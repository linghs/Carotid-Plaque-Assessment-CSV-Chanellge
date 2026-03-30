"""
Merge segmentation and classification predictions into final submission format.

Usage:
    python csv_merge_predictions.py --seg-dir ./data/val/preds_seg --cls-dir ./data/val/preds_cls --output-dir ./data/val/preds
"""

import os
import sys
import glob
import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm


def merge_predictions(seg_dir: str, cls_dir: str, output_dir: str):
    """
    Merge segmentation and classification predictions
    
    Args:
        seg_dir: Directory containing segmentation predictions (long_mask, trans_mask)
        cls_dir: Directory containing classification predictions (cls)
        output_dir: Output directory for merged predictions
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all segmentation prediction files
    seg_files = sorted(glob.glob(os.path.join(seg_dir, "*_pred.h5")))
    
    if len(seg_files) == 0:
        raise FileNotFoundError(f"No prediction files found in {seg_dir}")
    
    print(f"Found {len(seg_files)} segmentation prediction files")
    
    merged_count = 0
    
    for seg_file in tqdm(seg_files, desc="Merging predictions"):
        basename = os.path.basename(seg_file)
        
        # Find corresponding classification file
        cls_file = os.path.join(cls_dir, basename)
        
        if not os.path.exists(cls_file):
            print(f"Warning: No classification file found for {basename}, skipping...")
            continue
        
        # Read segmentation predictions
        with h5py.File(seg_file, 'r') as f:
            long_mask = f['long_mask'][:] if 'long_mask' in f else None
            trans_mask = f['trans_mask'][:] if 'trans_mask' in f else None
        
        # Read classification prediction
        with h5py.File(cls_file, 'r') as f:
            cls_label = f['cls'][:]
        
        # Write merged prediction
        output_file = os.path.join(output_dir, basename)
        with h5py.File(output_file, 'w') as f:
            if long_mask is not None:
                f.create_dataset('long_mask', data=long_mask, compression='gzip')
            if trans_mask is not None:
                f.create_dataset('trans_mask', data=trans_mask, compression='gzip')
            f.create_dataset('cls', data=cls_label)
        
        merged_count += 1
    
    print(f"\nSuccessfully merged {merged_count} predictions")
    print(f"Output directory: {output_dir}")
    
    return merged_count


def main():
    parser = argparse.ArgumentParser(description="Merge segmentation and classification predictions")
    parser.add_argument("--seg-dir", type=str, required=True,
                        help="Directory containing segmentation predictions")
    parser.add_argument("--cls-dir", type=str, required=True,
                        help="Directory containing classification predictions")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for merged predictions (default: ./data/val/preds)")
    parser.add_argument("--create-archive", action="store_true",
                        help="Create tar.gz archive of merged predictions")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        # Default output directory
        seg_path = Path(args.seg_dir)
        args.output_dir = seg_path.parent / "preds"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge predictions
    merged_count = merge_predictions(args.seg_dir, args.cls_dir, str(output_dir))
    
    if merged_count == 0:
        print("No predictions were merged!")
        return
    
    # Create archive if requested
    if args.create_archive:
        import tarfile
        tar_path = output_dir.parent / "preds.tar.gz"
        print(f"\nCreating archive: {tar_path}")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(output_dir, arcname=output_dir.name)
        print(f"Archive created: {tar_path} ({tar_path.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()


