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

from csv_dataset import CSVDataset, get_csv_transforms
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
    tolerance: float = 2.0
) -> MetricsTracker:
    """
    Train for one epoch (Segmentation only)
    
    Args:
        tolerance: NSD tolerance for MetricsTracker
    
    Returns:
        MetricsTracker with training metrics
    """
    model.train()
    metrics_tracker = MetricsTracker(num_cls_classes=2, tolerance=tolerance)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        has_labels = batch['has_label']
        view_types = batch['view']  # 获取视图类型 ('long' 或 'trans')
        
        # Forward pass
        outputs = model(images)
        
        # Prepare targets (only segmentation)
        targets = {'masks': masks}
        
        # Compute loss
        loss, loss_dict = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics (only for labeled data)
        # Convert segmentation logits to masks with values {0, 128, 255}
        pred_seg_logits = outputs['seg_logits']
        pred_seg_masks = torch.argmax(pred_seg_logits, dim=1)  # [B, H, W] with values {0, 1, 2}
        
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
            'seg_loss': f"{loss_dict.get('seg_loss', 0):.4f}",
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
    Validate the model (Segmentation only) with detailed logging and optional visualization
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss criterion
        device: Device to run on
        tolerance: NSD tolerance for MetricsTracker
        logger: Optional logger for detailed output
        writer: Optional TensorBoard writer
        epoch: Current epoch number
        resize_target: Target size for model input
    
    Returns:
        MetricsTracker with validation metrics
    """
    model.eval()
    metrics_tracker = MetricsTracker(num_cls_classes=2, tolerance=tolerance)
    
    # Manual metrics tracking for detailed logging
    dice_long = {1: 0.0, 2: 0.0}  # 1: Plaque, 2: Vessel
    dice_trans = {1: 0.0, 2: 0.0}
    nsd_long = {1: 0.0, 2: 0.0}
    nsd_trans = {1: 0.0, 2: 0.0}
    
    num_long = 0
    num_trans = 0
    val_idx = 0
    
    pbar = tqdm(val_loader, desc="[Validation]")
    
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        view_types = batch['view']  # 获取视图类型
        
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
        
        # Prepare targets (only segmentation)
        targets = {'masks': masks}
        
        # Compute loss
        loss, loss_dict = criterion(outputs, targets)
        
        # Convert segmentation logits to masks
        pred_seg_logits = outputs['seg_logits']
        
        # Resize back to original size
        if pred_seg_logits.shape[-2:] != (h_orig, w_orig):
            pred_seg_logits = F.interpolate(pred_seg_logits, (h_orig, w_orig), 
                                           mode="bilinear", align_corners=False)
        
        pred_seg_masks = torch.argmax(pred_seg_logits, dim=1)  # [B, H, W] with values {0, 1, 2}
        
        # Convert to {0, 128, 255}
        pred_seg_converted = torch.zeros_like(pred_seg_masks)
        pred_seg_converted[pred_seg_masks == 1] = 128  # Plaque
        pred_seg_converted[pred_seg_masks == 2] = 255  # Vessel
        
        # -------------------------
        # Visualization to TensorBoard (first sample of first batch)
        # -------------------------
        if writer is not None and val_idx == 0:
            try:
                # Get first sample
                img_vis = images[0, 0].cpu().numpy()
                pred_vis = pred_seg_masks[0].cpu().numpy()
                gt_vis = masks[0].cpu().numpy()
                
                # Convert gt from {0, 128, 255} to {0, 1, 2}
                gt_vis_indexed = np.zeros_like(gt_vis)
                gt_vis_indexed[gt_vis == 128] = 1  # Plaque
                gt_vis_indexed[gt_vis == 255] = 2  # Vessel
                
                # Normalize image
                def normalize_im(im):
                    mn = im.min()
                    mx = im.max()
                    if mx - mn < 1e-8:
                        return im - mn
                    return (im - mn) / (mx - mn)
                
                img_norm = normalize_im(img_vis)
                
                # RGB base
                base = np.stack([img_norm, img_norm, img_norm], axis=0)  # C,H,W
                
                def overlay(base_img, mask, color):
                    # base_img: C,H,W in [0,1], mask: H,W bool
                    over = base_img.copy()
                    alpha = 0.5
                    for c in range(3):
                        over[c][mask] = over[c][mask] * (1 - alpha) + color[c] * alpha
                    return over
                
                # colors: plaque (class 1) = red, vessel (class 2) = green
                red = [1.0, 0.0, 0.0]
                green = [0.0, 1.0, 0.0]
                
                pred_vis_rgb = base.copy()
                pred_vis_rgb = overlay(pred_vis_rgb, pred_vis == 1, red)
                pred_vis_rgb = overlay(pred_vis_rgb, pred_vis == 2, green)
                
                gt_vis_rgb = base.copy()
                gt_vis_rgb = overlay(gt_vis_rgb, gt_vis_indexed == 1, red)
                gt_vis_rgb = overlay(gt_vis_rgb, gt_vis_indexed == 2, green)
                
                # Concatenate horizontally: [C, H, W*3]
                concat_vis = np.concatenate([base, pred_vis_rgb, gt_vis_rgb], axis=2)
                
                view_type_str = view_types[0] if isinstance(view_types, (list, tuple)) else str(view_types)
                tag = f"Val/vis/{view_type_str}/sample_0"
                step = epoch if epoch is not None and epoch >= 0 else 0
                writer.add_image(tag, torch.from_numpy(concat_vis).float(), global_step=step)
            except Exception as e:
                if logger is not None:
                    logger.warning(f"Failed to write val visualization: {e}")
        
        # -------------------------
        # Calculate metrics per sample
        # -------------------------
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
            
            # Calculate per-class metrics for detailed logging
            for cls_idx, cls_value in enumerate([128, 255], start=1):  # 1: Plaque, 2: Vessel
                dsc, nsd = calculate_dsc_nsd_for_multiclass(
                    pred_single[0], gt_single[0], 
                    class_value=cls_value, 
                    tolerance=tolerance
                )
                
                if view_type == 'long':
                    dice_long[cls_idx] += dsc
                    nsd_long[cls_idx] += nsd
                else:  # trans
                    dice_trans[cls_idx] += dsc
                    nsd_trans[cls_idx] += nsd
            
            if view_type == 'long':
                num_long += 1
            else:
                num_trans += 1
        
        val_idx += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}"
        })
    
    # -------------------------
    # Detailed logging (similar to reference code)
    # -------------------------
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
    
    # -------------------------
    # TensorBoard logging
    # -------------------------
    if writer is not None and epoch is not None:
        all_metrics = metrics_tracker.get_all_metrics()
        writer.add_scalar('Val/Loss', all_metrics.get('avg_loss', 0), epoch)
        writer.add_scalar('Val/Segmentation_Score', all_metrics.get('segmentation_score', 0), epoch)
        
        # Per-class metrics
        for view in ['long', 'trans']:
            for cls_name in ['plaque', 'vessel']:
                key_dsc = f'{view}_dsc_{cls_name}'
                key_nsd = f'{view}_nsd_{cls_name}'
                if key_dsc in all_metrics:
                    writer.add_scalar(f'Val/DSC_{view}_{cls_name}', all_metrics[key_dsc], epoch)
                if key_nsd in all_metrics:
                    writer.add_scalar(f'Val/NSD_{view}_{cls_name}', all_metrics[key_nsd], epoch)
    
    return metrics_tracker


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    val_indices_cls0 = cls_0_cases[:25]
    val_indices_cls1 = cls_1_cases[:25]
    val_indices = sorted(val_indices_cls0 + val_indices_cls1)
    
    train_indices_cls0 = cls_0_cases[25:]
    train_indices_cls1 = cls_1_cases[25:]
    train_indices = sorted(train_indices_cls0 + train_indices_cls1)
    
    print(f"Train: {len(train_indices)} cases, Val: {len(val_indices)} cases")
    
    # Create datasets
    train_dataset = CSVDataset(
        data_root=args.data_root,
        split='train',
        view=args.view,
        transforms=get_csv_transforms(is_train=True, image_size=args.image_size),
        train_indices=train_indices,
        val_indices=val_indices,
        use_unlabeled=args.use_unlabeled
    )
    
    val_dataset = CSVDataset(
        data_root=args.data_root,
        split='val',
        view=args.view,
        transforms=get_csv_transforms(is_train=False, image_size=args.image_size),
        train_indices=train_indices,
        val_indices=val_indices,
        use_unlabeled=False
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
        #batch_size=args.batch_size,
        batch_size=1,
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
        use_classification=False  # Disable classification
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss without classification
    criterion = CSVLoss(
        seg_weight=args.seg_weight,
        cls_weight=0.0,  # No classification loss
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
    print("Training started (Segmentation Only)...")
    
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
            tolerance=args.nsd_tolerance
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
    
    print(f"\nTraining completed!")
    print(f"Best Segmentation Score: {best_val_score:.4f} at Epoch {best_epoch}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CSV Segmentation Model')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='./data/train',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='./csv_seg_outputs',
                        help='Output directory for checkpoints')
    parser.add_argument('--view', type=str, default='both', choices=['long', 'trans', 'both'],
                        help='Which view to use')
    parser.add_argument('--use_unlabeled', action='store_true',
                        help='Use unlabeled data for training')
    
    # Model arguments
    parser.add_argument('--encoder', type=str, default='efficientnet-b4',
                        help='Encoder backbone')
    parser.add_argument('--encoder_weights', type=str, default='imagenet',
                        help='Encoder pretrained weights')
    parser.add_argument('--num_seg_classes', type=int, default=3,
                        help='Number of segmentation classes (0: background, 1: plaque, 2: vessel)')
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
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--nsd_tolerance', type=float, default=2.0,
                        help='NSD (Normalized Surface Distance) tolerance in mm (default: 2.0)')
    
    args = parser.parse_args()
    
    main(args)


