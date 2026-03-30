"""
CSV 2026 Challenge - Two-Stage Prediction Pipeline
Stage 1: Segmentation - predict masks from images
Stage 2: Classification - predict class from predicted masks
"""

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
from csv_model_cls import CSVClassificationModel


class ValH5Dataset:
    """Simple dataset to iterate over val image .h5 files"""
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
    
    ckpt = torch.load(ckpt_path, map_location=device)
    
    if isinstance(ckpt, dict):
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
    print(f"✓ Loaded checkpoint: {ckpt_path}")
    return model


def predict_segmentation(seg_model, device, long_t, trans_t, long_shape, trans_shape, resize_target):
    """
    Stage 1: Predict segmentation masks
    
    Returns:
        long_mask: [H, W] numpy array with values {0, 128, 255}
        trans_mask: [H, W] numpy array with values {0, 128, 255}
    """
    seg_model.eval()
    
    with torch.no_grad():
        # Process long axis view
        xL = long_t.unsqueeze(0).to(device)  # [1, 3, H, W]
        xL_r = F.interpolate(xL, (resize_target, resize_target), mode="bilinear", align_corners=False)
        outputs_L = seg_model(xL_r)
        segL_logits = outputs_L['seg_logits']
        segL_up = F.interpolate(segL_logits, long_shape, mode="bilinear", align_corners=False)
        predL = torch.argmax(segL_up, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        
        # Convert from {0,1,2} to {0,128,255}
        predL_converted = np.zeros_like(predL)
        predL_converted[predL == 1] = 128  # Class 1
        predL_converted[predL == 2] = 255  # Class 2
        
        # Process transverse view
        xT = trans_t.unsqueeze(0).to(device)  # [1, 3, H, W]
        xT_r = F.interpolate(xT, (resize_target, resize_target), mode="bilinear", align_corners=False)
        outputs_T = seg_model(xT_r)
        segT_logits = outputs_T['seg_logits']
        segT_up = F.interpolate(segT_logits, trans_shape, mode="bilinear", align_corners=False)
        predT = torch.argmax(segT_up, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        
        # Convert from {0,1,2} to {0,128,255}
        predT_converted = np.zeros_like(predT)
        predT_converted[predT == 1] = 128  # Class 1
        predT_converted[predT == 2] = 255  # Class 2
    
    return predL_converted, predT_converted


def predict_classification(cls_model, device, long_mask, trans_mask, resize_target):
    """
    Stage 2: Predict classification from masks
    
    Args:
        long_mask: [H, W] numpy array with values {0, 128, 255}
        trans_mask: [H, W] numpy array with values {0, 128, 255}
    
    Returns:
        cls_pred: int, predicted class
        cls_prob: numpy array, class probabilities
    """
    cls_model.eval()
    
    with torch.no_grad():
        # Convert masks to tensors and normalize to [0, 1]
        long_mask_t = torch.from_numpy(long_mask).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        trans_mask_t = torch.from_numpy(trans_mask).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Normalize to [0, 1]
        long_mask_t = long_mask_t / 255.0
        trans_mask_t = trans_mask_t / 255.0
        
        # Resize to target size
        long_mask_t = F.interpolate(long_mask_t, (resize_target, resize_target), mode="bilinear", align_corners=False)
        trans_mask_t = F.interpolate(trans_mask_t, (resize_target, resize_target), mode="bilinear", align_corners=False)
        
        # Move to device
        long_mask_t = long_mask_t.to(device)
        trans_mask_t = trans_mask_t.to(device)
        
        # Forward pass
        outputs = cls_model(long_mask_t, trans_mask_t)
        cls_logits = outputs['cls_logits']
        
        # Get classification prediction
        cls_prob = torch.softmax(cls_logits, dim=-1).cpu().numpy()[0]
        cls_pred = np.argmax(cls_prob)
    
    return cls_pred, cls_prob


def two_stage_predict_and_save(seg_model, cls_model, device, file_path, long_t, trans_t, 
                                long_shape, trans_shape, resize_target, out_dir, 
                                save_masks=True):
    """
    Run two-stage prediction and save results
    
    Args:
        seg_model: Segmentation model
        cls_model: Classification model
        device: torch device
        file_path: Input file path
        long_t, trans_t: Input image tensors
        long_shape, trans_shape: Original image shapes
        resize_target: Target size for model input
        out_dir: Output directory
        save_masks: Whether to save predicted masks
    
    Returns:
        out_path: Output file path
        cls_pred: Predicted class
        cls_prob: Class probabilities
    """
    os.makedirs(out_dir, exist_ok=True)
    
    basename = os.path.basename(file_path)
    name_no_ext = os.path.splitext(basename)[0]
    out_path = os.path.join(out_dir, f"{name_no_ext}_pred.h5")
    
    # Stage 1: Predict segmentation masks
    long_mask, trans_mask = predict_segmentation(
        seg_model, device, long_t, trans_t, long_shape, trans_shape, resize_target
    )
    
    # Stage 2: Predict classification from masks
    cls_pred, cls_prob = predict_classification(
        cls_model, device, long_mask, trans_mask, resize_target
    )
    
    # Save results to h5
    with h5py.File(out_path, "w") as hf:
        # Always save classification
        hf.create_dataset("cls", data=np.array([cls_pred], dtype=np.uint8))
        
        # Optionally save masks
        if save_masks:
            hf.create_dataset("long_mask", data=long_mask, compression="gzip")
            hf.create_dataset("trans_mask", data=trans_mask, compression="gzip")
    
    return out_path, cls_pred, cls_prob


def main():
    parser = argparse.ArgumentParser(
        description="CSV 2026 Challenge - Two-Stage Prediction (Seg -> Cls)"
    )
    
    # Data arguments
    parser.add_argument("--val-dir", type=str, default="./data/val",
                        help="Path to validation data directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: val-dir/preds_two_stage)")
    
    # Model checkpoints
    parser.add_argument("--seg-checkpoint", type=str, required=True,
                        help="Path to segmentation model checkpoint")
    parser.add_argument("--cls-checkpoint", type=str, required=True,
                        help="Path to classification model checkpoint")
    
    # Segmentation model arguments
    parser.add_argument("--seg-encoder", type=str, default="efficientnet-b4",
                        help="Segmentation encoder backbone")
    parser.add_argument("--num-seg-classes", type=int, default=3,
                        help="Number of segmentation classes")
    
    # Classification model arguments
    parser.add_argument("--cls-encoder", type=str, default="efficientnet-b4",
                        help="Classification encoder backbone (not used in mask-only model)")
    parser.add_argument("--num-cls-classes", type=int, default=2,
                        help="Number of classification classes")
    parser.add_argument("--fusion-method", type=str, default="concat",
                        choices=['concat', 'add', 'max', 'attention'],
                        help="Fusion method (not used in mask-only model)")
    
    # Inference arguments
    parser.add_argument("--resize-target", type=int, default=512,
                        help="Target size for model input")
    parser.add_argument("--save-masks", action="store_true", default=True,
                        help="Save predicted masks in output files")
    parser.add_argument("--no-save-masks", dest="save_masks", action="store_false",
                        help="Do not save predicted masks")
    parser.add_argument("--device", type=str, default="cuda", choices=['cuda', 'cpu'],
                        help="Device to use")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Warning: CUDA not available, using CPU")
    else:
        device = torch.device(args.device)
    
    print(f"Device: {device}")
    
    # Setup paths
    val_dir = Path(args.val_dir)
    images_dir = val_dir / "images"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    if args.output_dir is None:
        out_dir = val_dir / "preds_two_stage"
    else:
        out_dir = Path(args.output_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    
    # Load dataset
    ds = ValH5Dataset(str(images_dir))
    print(f"Found {len(ds)} files")
    
    if len(ds) == 0:
        print("No h5 files found!")
        return
    
    print("\n" + "="*80)
    print("STAGE 1: Loading Segmentation Model")
    print("="*80)
    
    # Load segmentation model
    seg_model = CSVModel(
        encoder_name=args.seg_encoder,
        encoder_weights=None,
        num_seg_classes=args.num_seg_classes,
        num_cls_classes=2,
        use_classification=False
    ).to(device)
    
    seg_model = load_checkpoint(seg_model, args.seg_checkpoint, device)
    
    print("\n" + "="*80)
    print("STAGE 2: Loading Classification Model (Mask-Only)")
    print("="*80)
    
    # Load classification model (mask-only version)
    cls_model = CSVClassificationModel(
        encoder_name=args.cls_encoder,  # Not used in mask-only model
        encoder_weights=None,
        num_cls_classes=args.num_cls_classes,
        fusion_method=args.fusion_method,  # Not used in mask-only model
        use_mask_features=True
    ).to(device)
    
    cls_model = load_checkpoint(cls_model, args.cls_checkpoint, device)
    
    print("\n" + "="*80)
    print("Running Two-Stage Inference Pipeline")
    print("="*80)
    print(f"Stage 1: Predict masks from images (seg model)")
    print(f"Stage 2: Predict class from masks (cls model)")
    print(f"Save masks: {args.save_masks}")
    print("="*80 + "\n")
    
    # Run inference
    predictions = []
    probabilities = []
    
    for idx in tqdm(range(len(ds)), desc="Processing"):
        p, long_t, trans_t, long_shape, trans_shape = ds[idx]
        
        out_path, cls_pred, cls_prob = two_stage_predict_and_save(
            seg_model=seg_model,
            cls_model=cls_model,
            device=device,
            file_path=p,
            long_t=long_t,
            trans_t=trans_t,
            long_shape=long_shape,
            trans_shape=trans_shape,
            resize_target=args.resize_target,
            out_dir=str(out_dir),
            save_masks=args.save_masks
        )
        
        predictions.append(cls_pred)
        probabilities.append(cls_prob)
    
    print(f"\n✓ Completed! Processed {len(ds)} files")
    print(f"✓ Output: {out_dir}")
    
    # Print prediction statistics
    unique, counts = np.unique(predictions, return_counts=True)
    print("\n" + "="*80)
    print("Classification Prediction Statistics:")
    print("="*80)
    for cls, count in zip(unique, counts):
        avg_prob = np.mean([probabilities[i][cls] for i in range(len(predictions)) 
                           if predictions[i] == cls])
        print(f"  Class {cls}: {count} ({count/len(predictions)*100:.1f}%) - Avg confidence: {avg_prob:.3f}")
    
    # Calculate average confidence
    avg_confidence = np.mean([np.max(prob) for prob in probabilities])
    print(f"\n  Overall average confidence: {avg_confidence:.3f}")
    print("="*80)
    
    # Create tar.gz archive
    print("\nCreating compressed archive...")
    import tarfile
    tar_path = out_dir.parent / "preds_two_stage.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.name)
    print(f"✓ Compressed: {tar_path} ({tar_path.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()


