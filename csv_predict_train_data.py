import os
import sys
import glob
import argparse
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

# Add current directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from csv_model import CSVModel


class ValH5Dataset:
    """Simple dataset to iterate over val image .h5 files.
    Each file is expected to contain 'long_img' and 'trans_img' numpy arrays.
    """
    def __init__(self, images_dir):
        self.images_dir = images_dir
        self.paths = sorted(glob.glob(os.path.join(images_dir, "*.h5")))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        with h5py.File(p, "r") as f:
            long_img = f["long_img"][:]
            trans_img = f["trans_img"][:]
        
        # Store original shapes
        long_shape = long_img.shape
        trans_shape = trans_img.shape
        
        # Convert to 3-channel (grayscale to RGB)
        long_img_3ch = np.stack([long_img, long_img, long_img], axis=-1).astype(np.float32)
        trans_img_3ch = np.stack([trans_img, trans_img, trans_img], axis=-1).astype(np.float32)
        
        # Normalize to [0, 255] if needed
        if long_img_3ch.max() <= 1.0:
            long_img_3ch = long_img_3ch * 255.0
        if trans_img_3ch.max() <= 1.0:
            trans_img_3ch = trans_img_3ch * 255.0
        
        long_img_3ch = long_img_3ch.astype(np.uint8)
        trans_img_3ch = trans_img_3ch.astype(np.uint8)
        
        # Convert to torch tensors: [3, H, W], float32, normalize with ImageNet stats
        long_t = torch.from_numpy(long_img_3ch.transpose(2, 0, 1)).float()
        trans_t = torch.from_numpy(trans_img_3ch.transpose(2, 0, 1)).float()
        
        # Normalize to [0, 1]
        long_t = long_t / 255.0
        trans_t = trans_t / 255.0
        
        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        long_t = (long_t - mean) / std
        trans_t = (trans_t - mean) / std
        
        return p, long_t, trans_t, long_shape, trans_shape


def load_checkpoint(model, ckpt_path, device):
    """Load model checkpoint"""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device,weights_only=False)
    
    if isinstance(ckpt, dict):
        # Common patterns: {"model_state_dict": ..., "state_dict": ..., "model": ...}
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt
    
    model.load_state_dict(state, strict=True)
    print(f"✓ Loaded checkpoint from {ckpt_path}")
    
    if isinstance(ckpt, dict) and 'epoch' in ckpt:
        print(f"  Checkpoint epoch: {ckpt['epoch']}")
        if 'best_val_dice' in ckpt:
            print(f"  Best val dice: {ckpt['best_val_dice']:.4f}")
    
    return model


def predict_and_save(model, device, file_path, long_t, trans_t, long_shape, trans_shape, 
                    resize_target, out_dir, view='both', use_classification=False):
    """Run prediction and save results to h5 file"""
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    
    basename = os.path.basename(file_path)
    name_no_ext = os.path.splitext(basename)[0]
    out_path = os.path.join(out_dir, f"{name_no_ext}_pred.h5")
    
    with torch.no_grad():
        # Process long axis view
        if view in ['long', 'both']:
            xL = long_t.unsqueeze(0).to(device)  # [1, 3, H, W]
            
            # Resize to target size (same as training)
            xL_r = F.interpolate(xL, (resize_target, resize_target), mode="bilinear", align_corners=False)
            
            # Forward pass
            outputs_L = model(xL_r)
            segL_logits = outputs_L['seg_logits']
            
            # Upsample logits back to original size
            segL_up = F.interpolate(segL_logits, long_shape, mode="bilinear", align_corners=False)
            
            # Get segmentation prediction
            predL = torch.argmax(segL_up, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            
            # Get classification prediction (if available and enabled)
            if use_classification and 'cls_logits' in outputs_L:
                cls_logits_L = outputs_L['cls_logits']
                cls_prob_L = torch.softmax(cls_logits_L, dim=-1).cpu().numpy()[0]
                cls_pred_L = np.argmax(cls_prob_L)
            else:
                cls_pred_L = 0
        
        # Process transverse view
        if view in ['trans', 'both']:
            xT = trans_t.unsqueeze(0).to(device)  # [1, 3, H, W]
            
            # Resize to target size (same as training)
            xT_r = F.interpolate(xT, (resize_target, resize_target), mode="bilinear", align_corners=False)
            
            # Forward pass
            outputs_T = model(xT_r)
            segT_logits = outputs_T['seg_logits']
            
            # Upsample logits back to original size
            segT_up = F.interpolate(segT_logits, trans_shape, mode="bilinear", align_corners=False)
            
            # Get segmentation prediction
            predT = torch.argmax(segT_up, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            
            # Get classification prediction (if available and enabled)
            if use_classification and 'cls_logits' in outputs_T:
                cls_logits_T = outputs_T['cls_logits']
                cls_prob_T = torch.softmax(cls_logits_T, dim=-1).cpu().numpy()[0]
                cls_pred_T = np.argmax(cls_prob_T)
            else:
                cls_pred_T = 0
        
        # Combine classification results (use max risk score)
        if view == 'both':
            cls_pred = max(cls_pred_L, cls_pred_T)
        elif view == 'long':
            cls_pred = cls_pred_L
        else:  # trans
            cls_pred = cls_pred_T
    
    # Save to h5 with same key names as label files: long_mask, trans_mask, cls
    with h5py.File(out_path, "w") as hf:
        if view in ['long', 'both']:
            hf.create_dataset("long_mask", data=predL, compression="gzip")
        if view in ['trans', 'both']:
            hf.create_dataset("trans_mask", data=predT, compression="gzip")
        hf.create_dataset("cls", data=np.array([cls_pred], dtype=np.uint8))
    
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Inference script for CSV 2026 Challenge (Carotid Plaque Segmentation)"
    )
    parser.add_argument(
        "--val-dir", 
        type=str, 
        default="./data/train",
        help="Path to validation folder that contains 'images' subfolder with .h5 files"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="./csv_outputs/best_model.pth", 
        help="Path to checkpoint (best_model.pth)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None, 
        help="Output directory (default: val_dir/preds)"
    )
    parser.add_argument(
        "--encoder", 
        type=str, 
        default="efficientnet-b4", 
        help="Encoder backbone (must match training)"
    )
    parser.add_argument(
        "--num-seg-classes", 
        type=int, 
        default=3, 
        help="Number of segmentation classes (0: background, 1: class1, 2: class2)"
    )
    parser.add_argument(
        "--num-cls-classes", 
        type=int, 
        default=2, 
        help="Number of classification classes"
    )
    parser.add_argument(
        "--use-classification", 
        action="store_true", 
        default=True,
        help="Enable classification head"
    )
    parser.add_argument(
        "--view", 
        type=str, 
        default="both", 
        choices=['long', 'trans', 'both'],
        help="Which view to predict"
    )
    parser.add_argument(
        "--resize-target", 
        type=int, 
        default=512, 
        help="Resize input size used during training"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda", 
        choices=['cuda', 'cpu'],
        help="Device to run inference on"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1, 
        help="Batch size for inference (currently only 1 is supported)"
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print("="*60)
    print("CSV 2026 Challenge - Prediction Script")
    print("="*60)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Setup paths
    val_dir = Path(args.val_dir)
    images_dir = val_dir / "images"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    if args.output_dir is None:
        out_dir = val_dir / "preds"
    else:
        out_dir = Path(args.output_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nInput directory: {images_dir}")
    print(f"Output directory: {out_dir}")
    
    # Create dataset
    ds = ValH5Dataset(str(images_dir))
    print(f"\nFound {len(ds)} h5 files to process")
    
    if len(ds) == 0:
        print("No h5 files found! Exiting.")
        return
    
    # Build model
    print("\n" + "="*60)
    print("Building model...")
    print("="*60)
    
    # Check if checkpoint has classification head
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    has_cls_head = any('classification_head' in key for key in state_dict.keys())
    
    if not has_cls_head:
        print("Note: Checkpoint does not contain classification head. Setting use_classification=False")
        use_classification = False
    else:
        use_classification = args.use_classification
    
    model = CSVModel(
        encoder_name=args.encoder,
        encoder_weights=None,  # We'll load from checkpoint
        num_seg_classes=args.num_seg_classes,
        num_cls_classes=args.num_cls_classes,
        use_classification=use_classification
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Load checkpoint
    print("\n" + "="*60)
    print("Loading checkpoint...")
    print("="*60)
    model = load_checkpoint(model, args.checkpoint, device)
    
    # Run inference
    print("\n" + "="*60)
    print("Running inference...")
    print("="*60)
    print(f"View mode: {args.view}")
    print(f"Resize target: {args.resize_target}")
    
    for idx in tqdm(range(len(ds)), desc="Processing"):
        p, long_t, trans_t, long_shape, trans_shape = ds[idx]
        out_path = predict_and_save(
            model=model,
            device=device,
            file_path=p,
            long_t=long_t,
            trans_t=trans_t,
            long_shape=long_shape,
            trans_shape=trans_shape,
            resize_target=args.resize_target,
            out_dir=str(out_dir),
            view=args.view,
            use_classification=use_classification
        )
    
    print("\n" + "="*60)
    print("Prediction completed!")
    print("="*60)
    print(f"Total files processed: {len(ds)}")
    print(f"Output directory: {out_dir}")
    print(f"\nPrediction files saved with format: <case_id>_pred.h5")
    print("Each file contains:")
    if args.view in ['long', 'both']:
        print("  - long_mask: Segmentation mask for longitudinal view")
    if args.view in ['trans', 'both']:
        print("  - trans_mask: Segmentation mask for transverse view")
    print("  - cls: Classification label (0: low risk, 1: high risk)")
    
    # Compress predictions
    print("\n" + "="*60)
    print("Compressing predictions...")
    print("="*60)
    
    import tarfile
    import time
    
    tar_path = out_dir.parent / "preds.tar.gz"
    
    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(out_dir, arcname=out_dir.name)
        
        tar_size = tar_path.stat().st_size / (1024 * 1024)  # MB
        print(f"✓ Compression successful!")
        print(f"  Compressed file: {tar_path}")
        print(f"  Size: {tar_size:.2f} MB")
        print(f"  Original directory: {out_dir}")
    except Exception as e:
        print(f"❌ Compression failed: {e}")
        print(f"  You can manually compress with:")
        print(f"  tar -czf {tar_path} -C {out_dir.parent} {out_dir.name}")


if __name__ == "__main__":
    main()

