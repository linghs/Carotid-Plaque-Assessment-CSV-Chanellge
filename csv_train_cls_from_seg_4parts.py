"""
训练脚本 - 从分割预测训练分类模型 (4部分输入，带膨胀)
使用分割模型的预测mask代替真实mask
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import argparse
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from csv_dataset_cls_from_seg_4parts import CSVClassificationFromSeg4Parts, get_csv_cls_from_seg_transforms
from csv_model_cls_4parts import CSVClassificationModel4Parts, CSVClassificationLoss


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries
    
    This ensures consistent results across multiple runs with the same seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set Python hash seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)


def worker_init_fn(worker_id):
    """
    Initialize random seed for each DataLoader worker
    
    This ensures reproducibility when using multiple workers in DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict:
    """Train for one epoch"""
    model.train()
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    # 使用 tqdm，确保在所有环境下都能看到进度
    pbar = tqdm(
        train_loader, 
        desc=f"Epoch {epoch} [Train]",
        ncols=100,  # 固定宽度
        file=sys.stdout,  # 确保输出到标准输出
        dynamic_ncols=False  # 禁用动态列宽
    )
    
    for batch_idx, batch in enumerate(pbar):
        # Get 4 image parts
        long_img_128 = batch['long_img_128'].to(device)
        long_img_255 = batch['long_img_255'].to(device)
        trans_img_128 = batch['trans_img_128'].to(device)
        trans_img_255 = batch['trans_img_255'].to(device)
        cls_labels = batch['cls_label'].to(device)
        
        # Forward pass with 4 inputs
        outputs = model(long_img_128, long_img_255, trans_img_128, trans_img_255)
        
        # Compute loss
        loss, loss_dict = criterion(outputs, cls_labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Get predictions
        pred_logits = outputs['cls_logits']
        pred_labels = torch.argmax(pred_logits, dim=1)
        
        all_predictions.extend(pred_labels.cpu().numpy().tolist())
        all_labels.extend(cls_labels.cpu().numpy().tolist())
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # 每10个batch打印一次（备用日志）
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            sys.stdout.write(f"\r  Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
            sys.stdout.flush()
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    valid_mask = all_labels != -1
    
    metrics = {}
    if valid_mask.any():
        metrics['accuracy'] = accuracy_score(all_labels[valid_mask], all_predictions[valid_mask])
        metrics['f1_macro'] = f1_score(all_labels[valid_mask], all_predictions[valid_mask], average='macro')
    metrics['avg_loss'] = total_loss / num_batches
    
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Validate the model"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    # 使用 tqdm，确保在所有环境下都能看到进度
    pbar = tqdm(
        val_loader, 
        desc="[Validation]",
        ncols=100,  # 固定宽度
        file=sys.stdout,  # 确保输出到标准输出
        dynamic_ncols=False  # 禁用动态列宽
    )
    
    for batch_idx, batch in enumerate(pbar):
        # Get 4 image parts
        long_img_128 = batch['long_img_128'].to(device)
        long_img_255 = batch['long_img_255'].to(device)
        trans_img_128 = batch['trans_img_128'].to(device)
        trans_img_255 = batch['trans_img_255'].to(device)
        cls_labels = batch['cls_label'].to(device)
        
        # Forward pass with 4 inputs
        outputs = model(long_img_128, long_img_255, trans_img_128, trans_img_255)
        
        # Compute loss
        loss, loss_dict = criterion(outputs, cls_labels)
        
        # Track metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Get predictions
        pred_logits = outputs['cls_logits']
        pred_labels = torch.argmax(pred_logits, dim=1)
        
        all_predictions.extend(pred_labels.cpu().numpy().tolist())
        all_labels.extend(cls_labels.cpu().numpy().tolist())
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # 每5个batch打印一次（备用日志）
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(val_loader):
            sys.stdout.write(f"\r  Batch [{batch_idx+1}/{len(val_loader)}] Loss: {loss.item():.4f}")
            sys.stdout.flush()
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    valid_mask = all_labels != -1
    
    metrics = {}
    if valid_mask.any():
        metrics['accuracy'] = accuracy_score(all_labels[valid_mask], all_predictions[valid_mask])
        metrics['f1_macro'] = f1_score(all_labels[valid_mask], all_predictions[valid_mask], average='macro')
        # Calculate per-class F1 scores
        f1_per_class = f1_score(all_labels[valid_mask], all_predictions[valid_mask], average=None)
        metrics['f1_class_0'] = f1_per_class[0]
        metrics['f1_class_1'] = f1_per_class[1]
        # classification_score = f1_class_0 * f1_class_1
        metrics['classification_score'] = f1_per_class[0] * f1_per_class[1]
    metrics['avg_loss'] = total_loss / num_batches
    
    return metrics


def main(args):
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Print reproducibility settings
    print(f"\n{'='*60}")
    print(f"Reproducibility Settings")
    print(f"{'='*60}")
    print(f"Random seed: {args.seed}")
    print(f"CUDA deterministic: {torch.backends.cudnn.deterministic if torch.cuda.is_available() else 'N/A'}")
    print(f"CUDA benchmark: {torch.backends.cudnn.benchmark if torch.cuda.is_available() else 'N/A'}")
    print(f"PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED', 'not set')}")
    print(f"{'='*60}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training CSV Classification From Seg (4 Parts)")
    print(f"{'='*60}")
    print(f"Data root: {args.data_root}")
    print(f"Seg predictions: {args.seg_pred_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Encoder: {args.encoder}")
    print(f"Fusion method: {args.fusion_method}")
    print(f"Dilation kernel: {args.dilation_kernel_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"{'='*60}\n")
    
    # *** Load labels and create stratified split (same as csv_train_cls.py) ***
    import h5py
    labels_dir = os.path.join(args.data_root, 'labels')
    cls_labels = {}
    for idx in range(200):
        label_path = os.path.join(labels_dir, f'{idx:04d}_label.h5')
        if os.path.exists(label_path):
            with h5py.File(label_path, 'r') as f:
                cls_labels[idx] = int(f['cls'][()])
    
    # Separate by class
    cls_0_cases = [idx for idx, label in cls_labels.items() if label == 0]
    cls_1_cases = [idx for idx, label in cls_labels.items() if label == 1]
    
    # Shuffle with seed for reproducibility
    np.random.seed(args.seed)
    np.random.shuffle(cls_0_cases)
    np.random.shuffle(cls_1_cases)
    
    # Stratified split: 50 of each class for validation
    val_indices_cls0 = cls_0_cases[:50]
    val_indices_cls1 = cls_1_cases[:50]
    val_indices = sorted(val_indices_cls0 + val_indices_cls1)
    
    train_indices_cls0 = cls_0_cases[50:]
    train_indices_cls1 = cls_1_cases[50:]
    train_indices = sorted(train_indices_cls0 + train_indices_cls1)
    
    print(f"\nDataset split (stratified):")
    print(f"  Train: {len(train_indices)} cases")
    print(f"    - Class 0: {len(train_indices_cls0)} cases")
    print(f"    - Class 1: {len(train_indices_cls1)} cases")
    print(f"  Val: {len(val_indices)} cases")
    print(f"    - Class 0: {len(val_indices_cls0)} cases")
    print(f"    - Class 1: {len(val_indices_cls1)} cases")
    print()
    
    # Create datasets with consistent split AND dilation
    train_transforms = get_csv_cls_from_seg_transforms(is_train=True, image_size=args.image_size)
    val_transforms = get_csv_cls_from_seg_transforms(is_train=False, image_size=args.image_size)
    
    train_dataset = CSVClassificationFromSeg4Parts(
        data_root=args.data_root,
        seg_pred_dir=args.seg_pred_dir,
        split='train',
        transforms=train_transforms,
        train_indices=train_indices,
        val_indices=val_indices,
        dilation_kernel_size=args.dilation_kernel_size
    )
    
    val_dataset = CSVClassificationFromSeg4Parts(
        data_root=args.data_root,
        seg_pred_dir=args.seg_pred_dir,
        split='val',
        transforms=val_transforms,
        train_indices=train_indices,
        val_indices=val_indices,
        dilation_kernel_size=args.dilation_kernel_size
    )
    
    # Create dataloaders with deterministic settings
    # Create PyTorch generator for reproducibility
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,  # Ensure worker reproducibility
        generator=g  # Use fixed generator for shuffling
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn  # Ensure worker reproducibility
    )
    
    # Create model
    model = CSVClassificationModel4Parts(
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights,
        num_cls_classes=args.num_cls_classes,
        fusion_method=args.fusion_method
    ).to(device)
    
    # Create loss
    criterion = CSVClassificationLoss()
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=args.min_lr
        )
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
    else:
        scheduler = None
    
    # Training loop
    best_val_score = 0.0
    best_epoch = 0
    
    print(f"\n{'='*60}")
    print(f"开始训练...")
    print(f"总轮数: {args.num_epochs}")
    print(f"训练集batch数: {len(train_loader)}")
    print(f"验证集batch数: {len(val_loader)}")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch [{epoch}/{args.num_epochs}]")
        print(f"{'='*60}")
        sys.stdout.flush()
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        print(f"\nTrain Loss: {train_metrics['avg_loss']:.4f}")
        if 'accuracy' in train_metrics:
            print(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
            print(f"Train F1 (macro): {train_metrics['f1_macro']:.4f}")
        sys.stdout.flush()
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        print(f"\nVal Loss: {val_metrics['avg_loss']:.4f}")
        if 'accuracy' in val_metrics:
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val F1 (macro): {val_metrics['f1_macro']:.4f}")
        sys.stdout.flush()
        
        val_cls_score = val_metrics.get('classification_score', 0.0)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_cls_score > best_val_score:
            best_val_score = val_cls_score
            best_epoch = epoch
            
            save_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_score': best_val_score,
                'args': vars(args)
            }, save_path)
            print(f"✓ New best model saved: F1 = {best_val_score:.4f} at epoch {best_epoch}")
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            save_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cls_score': val_cls_score,
                'args': vars(args)
            }, save_path)
        
        if scheduler is not None:
            scheduler.step()
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best F1 Score: {best_val_score:.4f} at Epoch {best_epoch}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train CSV Classification From Seg (4 Parts with Dilation)'
    )
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='./data/train',
                        help='Path to data directory')
    parser.add_argument('--seg_pred_dir', type=str, default='./csv_seg_semi_outputs/preds',
                        help='Path to segmentation predictions directory')
    parser.add_argument('--output_dir', type=str, default='./csv_cls_from_seg_4parts_outputs',
                        help='Output directory')
    
    # Model arguments
    parser.add_argument('--encoder', type=str, default='resnet18',
                        help='Encoder backbone')
    parser.add_argument('--encoder_weights', type=str, default='imagenet',
                        help='Encoder pretrained weights')
    parser.add_argument('--num_cls_classes', type=int, default=2,
                        help='Number of classification classes')
    parser.add_argument('--fusion_method', type=str, default='concat', 
                        choices=['concat', 'add', 'attention'],
                        help='Feature fusion method')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Input image size')
    
    # Dilation argument
    parser.add_argument('--dilation_kernel_size', type=int, default=5,
                        help='Kernel size for dilating masks (0=no dilation)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    
    # Scheduler arguments
    parser.add_argument('--scheduler', type=str, default='cosine', 
                        choices=['cosine', 'step', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=30,
                        help='Step size for StepLR')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Gamma for StepLR')
    
    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    main(args)
