"""
CSV 2026 Challenge - 4-Part Classification Prediction
预测脚本，使用4部分分类模型预测data/val的结果
基于分割预测mask生成4个输入部分
"""
import os
import sys
import glob
import argparse
import h5py
import numpy as np
import torch
import cv2
from tqdm import tqdm
from pathlib import Path

# Add current directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from csv_model_cls_4parts import CSVClassificationModel4Parts


class ValMaskDataset:
    """Dataset to iterate over segmentation prediction .h5 files.
    Each file contains 'long_mask' and 'trans_mask' numpy arrays.
    For 4-part classification, we split masks into 4 separate parts:
    - long_mask with value==1 (class 128)
    - long_mask with value==2 (class 255)
    - trans_mask with value==1 (class 128)
    - trans_mask with value==2 (class 255)
    """
    def __init__(self, masks_dir, images_dir, dilation_kernel_size=5):
        self.masks_dir = masks_dir
        self.images_dir = images_dir
        self.dilation_kernel_size = dilation_kernel_size
        self.paths = sorted(glob.glob(os.path.join(masks_dir, "*.h5")))
        
        print(f"ValMaskDataset initialized:")
        print(f"  - Masks dir: {masks_dir}")
        print(f"  - Images dir: {images_dir}")
        print(f"  - Dilation kernel size: {dilation_kernel_size}")
        print(f"  - Found {len(self.paths)} mask files")

    def __len__(self):
        return len(self.paths)
    
    def _dilate_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Dilate mask to include surrounding pixels
        
        Args:
            mask: Binary mask [H, W] with 0/1 values
            
        Returns:
            Dilated mask [H, W]
        """
        if self.dilation_kernel_size <= 0:
            return mask
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.dilation_kernel_size, self.dilation_kernel_size)
        )
        dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        return dilated_mask

    def __getitem__(self, idx):
        mask_path = self.paths[idx]
        
        # Extract case_id from filename (e.g., "0000_pred.h5" -> "0000")
        basename = os.path.basename(mask_path)
        case_id = basename.replace("_pred.h5", "").replace(".h5", "")
        
        # Load segmentation prediction masks
        with h5py.File(mask_path, "r") as f:
            # Predicted masks: 0 (background), 1 (class 1), 2 (class 2)
            long_pred_mask = f["long_mask"][:].astype(np.uint8)
            trans_pred_mask = f["trans_mask"][:].astype(np.uint8)
        
        # Load original images
        image_path = os.path.join(self.images_dir, f"{case_id}.h5")
        with h5py.File(image_path, "r") as f:
            long_img = f["long_img"][:]
            trans_img = f["trans_img"][:]
        
        # Convert to 3-channel (grayscale to RGB)
        long_img = np.stack([long_img, long_img, long_img], axis=-1).astype(np.float32)
        trans_img = np.stack([trans_img, trans_img, trans_img], axis=-1).astype(np.float32)
        
        # Normalize to [0, 255] if needed
        if long_img.max() <= 1.0:
            long_img = long_img * 255.0
        if trans_img.max() <= 1.0:
            trans_img = trans_img * 255.0
        
        long_img = long_img.astype(np.uint8)
        trans_img = trans_img.astype(np.uint8)
        
        # *** CREATE 4 SEPARATE IMAGE PARTS WITH DILATED MASKS ***
        # Convert predicted masks: extract 128 and 255 value regions
        long_mask_128 = (long_pred_mask == 128).astype(np.uint8)  # Class 128
        long_mask_255 = (long_pred_mask == 255).astype(np.uint8)  # Class 255
        trans_mask_128 = (trans_pred_mask == 128).astype(np.uint8)
        trans_mask_255 = (trans_pred_mask == 255).astype(np.uint8)
        
        # Dilate masks to include surrounding pixels
        long_mask_128_dilated = self._dilate_mask(long_mask_128)
        long_mask_255_dilated = self._dilate_mask(long_mask_255)
        trans_mask_128_dilated = self._dilate_mask(trans_mask_128)
        trans_mask_255_dilated = self._dilate_mask(trans_mask_255)
        
        # Expand to 3 channels [H, W, 3]
        long_mask_128_3ch = np.stack([long_mask_128_dilated] * 3, axis=-1)
        long_mask_255_3ch = np.stack([long_mask_255_dilated] * 3, axis=-1)
        trans_mask_128_3ch = np.stack([trans_mask_128_dilated] * 3, axis=-1)
        trans_mask_255_3ch = np.stack([trans_mask_255_dilated] * 3, axis=-1)
        
        # Apply dilated masks to create 4 separate images
        long_img_128 = long_img * long_mask_128_3ch
        long_img_255 = long_img * long_mask_255_3ch
        trans_img_128 = trans_img * trans_mask_128_3ch
        trans_img_255 = trans_img * trans_mask_255_3ch
        
        # Normalize and convert to torch tensors [3, H, W]
        # Use ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        
        long_img_128 = (long_img_128.astype(np.float32) / 255.0 - mean) / std
        long_img_255 = (long_img_255.astype(np.float32) / 255.0 - mean) / std
        trans_img_128 = (trans_img_128.astype(np.float32) / 255.0 - mean) / std
        trans_img_255 = (trans_img_255.astype(np.float32) / 255.0 - mean) / std
        
        # Convert to torch tensors [3, H, W]
        long_t_128 = torch.from_numpy(long_img_128.transpose(2, 0, 1)).float()
        long_t_255 = torch.from_numpy(long_img_255.transpose(2, 0, 1)).float()
        trans_t_128 = torch.from_numpy(trans_img_128.transpose(2, 0, 1)).float()
        trans_t_255 = torch.from_numpy(trans_img_255.transpose(2, 0, 1)).float()
        
        return mask_path, long_t_128, long_t_255, trans_t_128, trans_t_255


def load_checkpoint(model, ckpt_path, device):
    """Load model checkpoint"""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
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


def predict_and_save(model, device, file_path, long_128, long_255, trans_128, trans_255, out_dir):
    """Run classification prediction using 4 image parts and save results to h5 file"""
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    
    basename = os.path.basename(file_path)
    # Extract case_id (e.g., "0000_pred.h5" -> "0000")
    case_id = basename.replace("_pred.h5", "").replace(".h5", "")
    out_path = os.path.join(out_dir, f"{case_id}_pred.h5")
    
    with torch.no_grad():
        # Add batch dimension and move to device
        long_x_128 = long_128.unsqueeze(0).to(device)  # [1, 3, H, W]
        long_x_255 = long_255.unsqueeze(0).to(device)  # [1, 3, H, W]
        trans_x_128 = trans_128.unsqueeze(0).to(device)  # [1, 3, H, W]
        trans_x_255 = trans_255.unsqueeze(0).to(device)  # [1, 3, H, W]
        
        # Forward pass with 4 image parts
        outputs = model(long_x_128, long_x_255, trans_x_128, trans_x_255)
        cls_logits = outputs['cls_logits']
        
        # Get classification prediction
        cls_prob = torch.softmax(cls_logits, dim=-1).cpu().numpy()[0]
        cls_pred = np.argmax(cls_prob)
    
    # Save to h5 (only classification)
    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("cls", data=np.array([cls_pred], dtype=np.uint8))
    
    return out_path, cls_pred


def main():
    parser = argparse.ArgumentParser(description="CSV 2026 Challenge - 4-Part Classification Prediction")
    parser.add_argument("--val-dir", type=str, default="./data/val",
                        help="Path to validation directory")
    parser.add_argument("--checkpoint", type=str, default="./csv_cls_from_seg_4parts_outputs/best_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: val-dir/preds_cls)")
    parser.add_argument("--masks-subdir", type=str, default="preds_seg",
                        help="Subdirectory containing segmentation mask predictions")
    parser.add_argument("--images-subdir", type=str, default="images",
                        help="Subdirectory containing original images")
    parser.add_argument("--num-cls-classes", type=int, default=2,
                        help="Number of classification classes")
    parser.add_argument("--encoder", type=str, default="resnet152",
                        help="Encoder backbone (must match training)")
    parser.add_argument("--fusion-method", type=str, default="concat", choices=['concat', 'add', 'attention'],
                        help="Feature fusion method (must match training)")
    parser.add_argument("--dilation-kernel-size", type=int, default=5,
                        help="Kernel size for dilating masks (must match training)")
    parser.add_argument("--device", type=str, default="cuda", choices=['cuda', 'cpu'],
                        help="Device to use")
    
    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("⚠️  CUDA not available, using CPU")
    else:
        device = torch.device(args.device)
    
    print(f"Device: {device}")
    
    val_dir = Path(args.val_dir)
    masks_dir = val_dir / args.masks_subdir
    images_dir = val_dir / args.images_subdir
    
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    if args.output_dir is None:
        out_dir = val_dir / "preds_cls"
    else:
        out_dir = Path(args.output_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"CSV 4-Part Classification Prediction")
    print(f"{'='*60}")
    print(f"Input masks: {masks_dir}")
    print(f"Input images: {images_dir}")
    print(f"Output: {out_dir}")
    print(f"Encoder: {args.encoder}")
    print(f"Fusion method: {args.fusion_method}")
    print(f"Dilation kernel size: {args.dilation_kernel_size}")
    print(f"{'='*60}\n")
    
    # Create dataset
    ds = ValMaskDataset(
        str(masks_dir), 
        str(images_dir),
        dilation_kernel_size=args.dilation_kernel_size
    )
    print(f"Found {len(ds)} samples")
    
    if len(ds) == 0:
        print("⚠️  No h5 mask files found!")
        return
    
    # Create model for 4-part classification
    model = CSVClassificationModel4Parts(
        encoder_name=args.encoder,
        encoder_weights=None,  # Will be loaded from checkpoint
        num_cls_classes=args.num_cls_classes,
        fusion_method=args.fusion_method
    ).to(device)
    
    model = load_checkpoint(model, args.checkpoint, device)
    
    print("\nRunning classification inference (4-part model)...")
    predictions = []
    for idx in tqdm(range(len(ds)), desc="Processing"):
        p, long_128, long_255, trans_128, trans_255 = ds[idx]
        out_path, cls_pred = predict_and_save(
            model=model,
            device=device,
            file_path=p,
            long_128=long_128,
            long_255=long_255,
            trans_128=trans_128,
            trans_255=trans_255,
            out_dir=str(out_dir)
        )
        predictions.append(cls_pred)
    
    print(f"\n{'='*60}")
    print(f"✓ Completed! Processed {len(ds)} files")
    print(f"Output directory: {out_dir}")
    print(f"{'='*60}")
    
    # Print prediction statistics
    unique, counts = np.unique(predictions, return_counts=True)
    print("\nPrediction statistics:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} ({count/len(predictions)*100:.1f}%)")
    
    # Create tar.gz archive
    import tarfile
    tar_path = out_dir.parent / "preds_cls.tar.gz"
    print(f"\nCreating compressed archive...")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.name)
    print(f"✓ Compressed: {tar_path} ({tar_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"\n{'='*60}")
    print("Prediction completed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
