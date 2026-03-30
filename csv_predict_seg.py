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
    For segmentation, we process each view separately.
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


def predict_and_save(model, device, file_path, long_t, trans_t, long_shape, trans_shape, 
                    resize_target, out_dir, view='both'):
    """Run segmentation prediction and save results to h5 file"""
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
            
            # Get segmentation prediction (convert from {0,1,2} to {0,128,255})
            predL = torch.argmax(segL_up, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            predL_converted = np.zeros_like(predL)
            predL_converted[predL == 1] = 128  # Plaque
            predL_converted[predL == 2] = 255  # Vessel
        
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
            
            # Get segmentation prediction (convert from {0,1,2} to {0,128,255})
            predT = torch.argmax(segT_up, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            predT_converted = np.zeros_like(predT)
            predT_converted[predT == 1] = 128  # Plaque
            predT_converted[predT == 2] = 255  # Vessel
    
    # Save to h5 (only masks, no classification)
    with h5py.File(out_path, "w") as hf:
        if view in ['long', 'both']:
            hf.create_dataset("long_mask", data=predL_converted, compression="gzip")
        if view in ['trans', 'both']:
            hf.create_dataset("trans_mask", data=predT_converted, compression="gzip")
    
    return out_path


def main():
    parser = argparse.ArgumentParser(description="CSV 2026 Challenge - Segmentation Prediction")
    parser.add_argument("--val-dir", type=str, default="./data/val")
    parser.add_argument("--checkpoint", type=str, default="./csv_seg_outputs/best_model.pth")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--encoder", type=str, default="efficientnet-b4")
    parser.add_argument("--num-seg-classes", type=int, default=3)
    parser.add_argument("--view", type=str, default="both", choices=['long', 'trans', 'both'])
    parser.add_argument("--resize-target", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda", choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Device: {device}")
    
    val_dir = Path(args.val_dir)
    images_dir = val_dir / "images"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    if args.output_dir is None:
        out_dir = val_dir / "preds_seg"
    else:
        out_dir = Path(args.output_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    
    ds = ValH5Dataset(str(images_dir))
    print(f"Found {len(ds)} files")
    
    if len(ds) == 0:
        print("No h5 files found!")
        return
    
    # Model without classification
    model = CSVModel(
        encoder_name=args.encoder,
        encoder_weights=None,
        num_seg_classes=args.num_seg_classes,
        num_cls_classes=2,
        use_classification=False
    ).to(device)
    
    model = load_checkpoint(model, args.checkpoint, device)
    
    print("Running segmentation inference...")
    for idx in tqdm(range(len(ds)), desc="Processing"):
        p, long_t, trans_t, long_shape, trans_shape = ds[idx]
        predict_and_save(
            model=model,
            device=device,
            file_path=p,
            long_t=long_t,
            trans_t=trans_t,
            long_shape=long_shape,
            trans_shape=trans_shape,
            resize_target=args.resize_target,
            out_dir=str(out_dir),
            view=args.view
        )
    
    print(f"\nCompleted! Processed {len(ds)} files")
    print(f"Output: {out_dir}")
    
    # Create tar.gz archive
    import tarfile
    tar_path = out_dir.parent / "preds_seg.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.name)
    print(f"Compressed: {tar_path} ({tar_path.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()


