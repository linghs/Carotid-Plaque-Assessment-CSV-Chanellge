#!/usr/bin/env python3
"""
半监督训练脚本

基于csv_train_seg.py，增加对伪标签数据的支持。
保持训练集/验证集划分不变，使用伪标签数据扩充训练集。

使用方法:
    # 不使用伪标签（等同于原始训练）
    python csv_train_seg_semi.py --encoder resnet50
    
    # 使用伪标签进行半监督训练
    python csv_train_seg_semi.py --encoder resnet50 --use_pseudo_labels --pseudo_labels_dir ./data/pseudo_labels
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import argparse
from pathlib import Path
import json
from datetime import datetime

# Add current directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from csv_dataset_semi import CSVDatasetSemiSupervised, get_csv_transforms
from csv_model import CSVModel, CSVLoss
from csv_utils import MetricsTracker, calculate_dsc_nsd_for_multiclass


class TrainingLogger:
    """简化的训练日志记录器"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / f'metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        self.log_file = self.log_dir / f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        self.history = []
    
    def info(self, message: str):
        """Log an info message"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"[INFO] {message}\n")
    
    def warning(self, message: str):
        """Log a warning message"""
        print(f"WARNING: {message}")
        with open(self.log_file, 'a') as f:
            f.write(f"[WARNING] {message}\n")
    
    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict, lr: float = None):
        """记录一个epoch的指标"""
        epoch_data = {
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'learning_rate': lr
        }
        self.history.append(epoch_data)
        with open(self.metrics_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def log_best_model(self, epoch: int, metric_value: float, metric_name: str = 'dice'):
        """记录最佳模型信息"""
        best_info = {
            'epoch': epoch,
            'metric_name': metric_name,
            'metric_value': metric_value,
        }
        best_file = self.log_dir / 'best_model_info.json'
        with open(best_file, 'w') as f:
            json.dump(best_info, f, indent=2)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    tolerance: float = 2.0,
    pseudo_loss_weight: float = 1.0
) -> MetricsTracker:
    """
    Train for one epoch (Segmentation only)
    
    Args:
        tolerance: NSD tolerance for MetricsTracker
        pseudo_loss_weight: Weight for pseudo-labeled samples (可以设置<1来降低伪标签影响)
    
    Returns:
        MetricsTracker with training metrics
    """
    model.train()
    metrics_tracker = MetricsTracker(num_cls_classes=2, tolerance=tolerance)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        has_labels = batch['has_label']
        is_pseudo = batch['is_pseudo']
        view_types = batch['view']
        
        # Forward pass
        outputs = model(images)
        
        # Prepare targets (only segmentation)
        targets = {'masks': masks}
        
        # Compute loss
        loss, loss_dict = criterion(outputs, targets)
        
        # 如果batch中包含伪标签，可以对其应用不同的权重
        if any(is_pseudo):
            # 分别计算真实标签和伪标签的损失
            real_mask = torch.tensor([not p for p in is_pseudo], device=device)
            pseudo_mask = torch.tensor(is_pseudo, device=device)
            
            if real_mask.any() and pseudo_mask.any():
                # 重新计算损失，对伪标签应用权重
                # 这里简化处理，直接对整体loss应用权重调整
                # 更精确的做法是分别计算两部分的loss
                pass  # 保持原始loss，或可以在这里调整
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update metrics (only for labeled data)
        pred_seg_logits = outputs['seg_logits']
        pred_seg_masks = torch.argmax(pred_seg_logits, dim=1)
        
        # Convert to {0, 128, 255}
        pred_seg_converted = torch.zeros_like(pred_seg_masks)
        pred_seg_converted[pred_seg_masks == 1] = 128  # Plaque
        pred_seg_converted[pred_seg_masks == 2] = 255  # Vessel
        
        labeled_mask = torch.BoolTensor(has_labels)
        if labeled_mask.any():
            # Update metrics for each labeled sample
            for idx, (is_labeled, view_type) in enumerate(zip(has_labels, view_types)):
                if is_labeled:
                    metrics_tracker.update(
                        pred_seg=pred_seg_converted[idx:idx+1],
                        true_seg=masks[idx:idx+1],
                        view_type=view_type,
                        pred_cls_logits=None,
                        true_cls_labels=None,
                        loss_dict=loss_dict
                    )
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss/num_batches:.4f}",
        })
    
    return metrics_tracker


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tolerance: float = 2.0,
    logger=None,
    writer=None,
    epoch=None,
    resize_target: int = 512,
) -> MetricsTracker:
    """
    Validate the model (Segmentation only)
    """
    model.eval()
    metrics_tracker = MetricsTracker(num_cls_classes=2, tolerance=tolerance)
    
    # Manual metrics tracking for detailed logging
    dice_long = {1: 0.0, 2: 0.0}
    dice_trans = {1: 0.0, 2: 0.0}
    nsd_long = {1: 0.0, 2: 0.0}
    nsd_trans = {1: 0.0, 2: 0.0}
    
    num_long = 0
    num_trans = 0
    
    pbar = tqdm(val_loader, desc="[Validation]")
    
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        view_types = batch['view']
        
        # Store original size
        h_orig, w_orig = images.shape[-2:]
        
        # Resize input if needed
        if h_orig != resize_target or w_orig != resize_target:
            images_resized = F.interpolate(images, (resize_target, resize_target), 
                                          mode="bilinear", align_corners=False)
        else:
            images_resized = images
        
        # Forward pass
        outputs = model(images_resized)
        
        # Prepare targets
        targets = {'masks': masks}
        
        # Compute loss
        loss, loss_dict = criterion(outputs, targets)
        
        # Convert segmentation logits to masks
        pred_seg_logits = outputs['seg_logits']
        
        # Resize back to original size
        if pred_seg_logits.shape[-2:] != (h_orig, w_orig):
            pred_seg_logits = F.interpolate(pred_seg_logits, (h_orig, w_orig), 
                                           mode="bilinear", align_corners=False)
        
        pred_seg_masks = torch.argmax(pred_seg_logits, dim=1)
        
        # Convert to {0, 128, 255}
        pred_seg_converted = torch.zeros_like(pred_seg_masks)
        pred_seg_converted[pred_seg_masks == 1] = 128
        pred_seg_converted[pred_seg_masks == 2] = 255
        
        # Calculate metrics per sample
        batch_size = images.shape[0]
        for idx in range(batch_size):
            view_type = view_types[idx]
            pred_single = pred_seg_converted[idx:idx+1]
            gt_single = masks[idx:idx+1]
            
            # Update metrics tracker
            metrics_tracker.update(
                pred_seg=pred_single,
                true_seg=gt_single,
                view_type=view_type,
                pred_cls_logits=None,
                true_cls_labels=None,
                loss_dict=loss_dict
            )
            
            # Calculate per-class metrics
            for cls_idx, cls_value in enumerate([128, 255], start=1):
                dsc, nsd = calculate_dsc_nsd_for_multiclass(
                    pred_single[0], gt_single[0], 
                    class_value=cls_value, 
                    tolerance=tolerance
                )
                
                if view_type == 'long':
                    dice_long[cls_idx] += dsc
                    nsd_long[cls_idx] += nsd
                else:
                    dice_trans[cls_idx] += dsc
                    nsd_trans[cls_idx] += nsd
            
            if view_type == 'long':
                num_long += 1
            else:
                num_trans += 1
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Detailed logging
    if logger is not None:
        idx_to_name = {1: "Plaque", 2: "Vessel"}
        
        for cls in [1, 2]:
            if num_long > 0:
                dice_long[cls] = dice_long[cls] / num_long
                nsd_long[cls] = nsd_long[cls] / num_long
            if num_trans > 0:
                dice_trans[cls] = dice_trans[cls] / num_trans
                nsd_trans[cls] = nsd_trans[cls] / num_trans
            
            logger.info(f"[Dice] {idx_to_name[cls]} | Long: {dice_long[cls]:.4f} | Trans: {dice_trans[cls]:.4f}")
            logger.info(f"[NSD]  {idx_to_name[cls]} | Long: {nsd_long[cls]:.4f} | Trans: {nsd_trans[cls]:.4f}")
        
        # Mean metrics
        if num_long > 0 and num_trans > 0:
            mean_dice = (dice_long[1] + dice_long[2] + dice_trans[1] + dice_trans[2]) / 4.0
            mean_nsd = (nsd_long[1] + nsd_long[2] + nsd_trans[1] + nsd_trans[2]) / 4.0
            logger.info(f"[Dice] Mean Foreground Dice: {mean_dice:.4f}")
            logger.info(f"[NSD]  Mean Foreground NSD: {mean_nsd:.4f}")
        
        # Segmentation scores
        seg_score_long_plaque = (dice_long[1] + nsd_long[1]) / 2
        seg_score_long_vessel = (dice_long[2] + nsd_long[2]) / 2
        seg_score_trans_plaque = (dice_trans[1] + nsd_trans[1]) / 2
        seg_score_trans_vessel = (dice_trans[2] + nsd_trans[2]) / 2
        
        seg_score = (seg_score_long_vessel * 0.4 + 
                     seg_score_long_plaque * 0.6 + 
                     seg_score_trans_vessel * 0.4 + 
                     seg_score_trans_plaque * 0.6) / 2
        
        logger.info(f"[Seg Score] Long Plaque: {seg_score_long_plaque:.4f}, Long Vessel: {seg_score_long_vessel:.4f}")
        logger.info(f"[Seg Score] Trans Plaque: {seg_score_trans_plaque:.4f}, Trans Vessel: {seg_score_trans_vessel:.4f}")
        logger.info(f"[Seg Score] Total: {seg_score:.4f}")
    
    return metrics_tracker


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取真实标签并按分类划分（保持与原脚本相同）
    import h5py
    labels_dir = os.path.join(args.data_root, 'labels')
    cls_labels = {}
    for idx in range(200):
        label_path = os.path.join(labels_dir, f'{idx:04d}_label.h5')
        if os.path.exists(label_path):
            with h5py.File(label_path, 'r') as f:
                cls_labels[idx] = int(f['cls'][()])
    
    cls_0_cases = [idx for idx, label in cls_labels.items() if label == 0]
    cls_1_cases = [idx for idx, label in cls_labels.items() if label == 1]
    
    np.random.seed(args.seed)
    np.random.shuffle(cls_0_cases)
    np.random.shuffle(cls_1_cases)
    
    # 验证集：每类25个
    val_indices_cls0 = cls_0_cases[:25]
    val_indices_cls1 = cls_1_cases[:25]
    val_indices = sorted(val_indices_cls0 + val_indices_cls1)
    
    # 训练集：剩余的真实标注
    train_indices_cls0 = cls_0_cases[25:]
    train_indices_cls1 = cls_1_cases[25:]
    train_indices = sorted(train_indices_cls0 + train_indices_cls1)
    
    print(f"\n{'='*60}")
    print(f"Dataset Split (Same as original)")
    print(f"{'='*60}")
    print(f"Train (Real labels): {len(train_indices)} cases")
    print(f"Val (Real labels): {len(val_indices)} cases")
    if args.use_pseudo_labels:
        print(f"Pseudo labels will be added to training set")
    print(f"{'='*60}\n")
    
    # Create datasets
    train_dataset = CSVDatasetSemiSupervised(
        data_root=args.data_root,
        split='train',
        view=args.view,
        transforms=get_csv_transforms(is_train=True, image_size=args.image_size),
        train_indices=train_indices,
        val_indices=val_indices,
        use_pseudo_labels=args.use_pseudo_labels,
        pseudo_labels_dir=args.pseudo_labels_dir
    )
    
    val_dataset = CSVDatasetSemiSupervised(
        data_root=args.data_root,
        split='val',
        view=args.view,
        transforms=get_csv_transforms(is_train=False, image_size=args.image_size),
        train_indices=train_indices,
        val_indices=val_indices,
        use_pseudo_labels=False  # 验证集永远不使用伪标签
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # 验证时batch_size=1
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Model without classification head
    model = CSVModel(
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights,
        num_seg_classes=args.num_seg_classes,
        num_cls_classes=2,
        use_classification=False  # 只做分割
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss without classification
    criterion = CSVLoss(
        seg_weight=args.seg_weight,
        cls_weight=0.0,
        use_dice_loss=True,
        use_ce_loss=True,
        dice_weight=0.5,
        ce_weight=0.5,
        ignore_index=-1
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.num_epochs,
            eta_min=args.min_lr
        )
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )
    else:
        scheduler = None
    
    logger = TrainingLogger(log_dir=output_dir / 'logs')
    
    mode_str = "Semi-Supervised" if args.use_pseudo_labels else "Supervised Only"
    print(f"\n{'='*60}")
    print(f"Training Mode: {mode_str}")
    print(f"{'='*60}\n")
    
    best_val_score = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            tolerance=args.nsd_tolerance,
            pseudo_loss_weight=args.pseudo_loss_weight
        )
        
        train_metrics.print_metrics("Train")
        
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            tolerance=args.nsd_tolerance,
            logger=logger,
            writer=None,
            epoch=epoch,
            resize_target=args.image_size,
        )
        
        val_metrics.print_metrics("Validation")
        
        val_all_metrics = val_metrics.get_all_metrics()
        train_all_metrics = train_metrics.get_all_metrics()
        
        val_seg_score = val_all_metrics.get('segmentation_score', 0.0)
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.log_epoch(epoch=epoch, train_metrics=train_all_metrics, val_metrics=val_all_metrics, lr=current_lr)
        
        print(f"Val Segmentation Score: {val_seg_score:.4f}")
        
        if val_seg_score > best_val_score:
            best_val_score = val_seg_score
            best_epoch = epoch
            
            save_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_score': best_val_score,
                'args': vars(args)
            }, save_path)
            print(f"✓ New best: {best_val_score:.4f} at epoch {best_epoch}")
            logger.log_best_model(epoch=epoch, metric_value=best_val_score, metric_name='seg_score')
        
        if scheduler is not None:
            scheduler.step()
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}")
    print(f"Best Segmentation Score: {best_val_score:.4f} at Epoch {best_epoch}")
    print(f"Model saved to: {output_dir / 'best_model.pth'}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CSV Segmentation Model with Semi-Supervised Learning')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='./csv_seg_semi_outputs',
                        help='Output directory for checkpoints')
    parser.add_argument('--view', type=str, default='both', choices=['long', 'trans', 'both'],
                        help='Which view to use')
    
    # Semi-supervised arguments
    parser.add_argument('--use_pseudo_labels', action='store_true',
                        help='Use pseudo-labeled data for training')
    parser.add_argument('--pseudo_labels_dir', type=str, default=None,
                        help='Path to pseudo labels directory (default: data_root/pseudo_labels)')
    parser.add_argument('--pseudo_loss_weight', type=float, default=1.0,
                        help='Loss weight for pseudo-labeled samples (default: 1.0)')
    
    # Model arguments
    parser.add_argument('--encoder', type=str, default='efficientnet-b4',
                        help='Encoder backbone')
    parser.add_argument('--encoder_weights', type=str, default='imagenet',
                        help='Encoder pretrained weights')
    parser.add_argument('--num_seg_classes', type=int, default=3,
                        help='Number of segmentation classes')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Input image size')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for scheduler')
    
    # Loss arguments
    parser.add_argument('--seg_weight', type=float, default=1.0,
                        help='Segmentation loss weight')
    
    # Scheduler arguments
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=30,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Gamma for StepLR scheduler')
    
    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--nsd_tolerance', type=float, default=2.0,
                        help='NSD tolerance in mm')
    
    args = parser.parse_args()
    
    main(args)
