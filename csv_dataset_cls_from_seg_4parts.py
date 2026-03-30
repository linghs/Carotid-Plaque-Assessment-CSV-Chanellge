import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional
import albumentations as A
import cv2


class CSVClassificationFromSeg4Parts(Dataset):
    """
    Dataset for CSV Classification - Uses segmentation predictions split into 4 parts
    
    Each sample contains 4 image parts based on predicted mask values:
    1. long_img with pred_mask==128 + surrounding area
    2. long_img with pred_mask==255 + surrounding area
    3. trans_img with pred_mask==128 + surrounding area
    4. trans_img with pred_mask==255 + surrounding area
    
    Key difference from csv_dataset_cls_4parts.py:
    - Uses predicted masks from seg_pred_dir instead of ground truth masks
    - Predicted masks format: 0 (background), 1 (class 1), 2 (class 2)
    
    Args:
        data_root: Path to data directory containing 'images/' and 'labels/'
        seg_pred_dir: Path to segmentation predictions directory
        split: 'train' or 'val'
        transforms: Albumentations transforms
        train_indices: List of training indices
        val_indices: List of validation indices
        dilation_kernel_size: Kernel size for dilating masks (default: 5)
    """
    
    def __init__(
        self, 
        data_root: str,
        seg_pred_dir: str,
        split: str = 'train',
        transforms: Optional[A.Compose] = None,
        train_indices: Optional[list] = None,
        val_indices: Optional[list] = None,
        dilation_kernel_size: int = 5,
    ):
        super().__init__()
        self.data_root = data_root
        self.seg_pred_dir = seg_pred_dir
        self.split = split
        self.transforms = transforms
        self.dilation_kernel_size = dilation_kernel_size
        
        self.images_dir = os.path.join(data_root, 'images')
        self.labels_dir = os.path.join(data_root, 'labels')
        
        if not os.path.exists(self.labels_dir):
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")
        
        if not os.path.exists(self.seg_pred_dir):
            raise FileNotFoundError(f"Segmentation predictions directory not found: {self.seg_pred_dir}")
        
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        # Build dataset samples
        self.samples = []
        
        # Split labeled data into train/val
        if train_indices is None or val_indices is None:
            # Default split: 160 train, 40 val
            np.random.seed(42)
            perm = np.random.permutation(200)
            train_indices = sorted(perm[:160].tolist())
            val_indices = sorted(perm[160:].tolist())
        
        if split == 'train':
            indices = train_indices
        else:  # val
            indices = val_indices
        
        # Verify that segmentation predictions exist
        missing_preds = []
        for idx in indices:
            pred_path = os.path.join(self.seg_pred_dir, f'{idx:04d}_pred.h5')
            if not os.path.exists(pred_path):
                missing_preds.append(idx)
        
        if missing_preds:
            print(f"Warning: Missing segmentation predictions for {len(missing_preds)} cases: {missing_preds[:10]}...")
            print(f"These cases will be skipped.")
            indices = [idx for idx in indices if idx not in missing_preds]
        
        # Each sample is a patient with both views
        for idx in indices:
            self.samples.append({
                'case_id': idx,
                'has_label': True
            })
        
        print(f"CSV Classification From Seg (4 Parts) [{split}] initialized:")
        print(f"  - Segmentation predictions: {seg_pred_dir}")
        print(f"  - Total samples: {len(self.samples)}")
        print(f"  - 4 parts: long_128, long_255, trans_128, trans_255")
        print(f"  - Dilation kernel size: {dilation_kernel_size}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
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
    
    def __getitem__(self, idx: int) -> dict:
        sample_info = self.samples[idx]
        case_id = sample_info['case_id']
        has_label = sample_info['has_label']
        
        # Load segmentation predictions (predicted masks)
        seg_pred_path = os.path.join(self.seg_pred_dir, f'{case_id:04d}_pred.h5')
        with h5py.File(seg_pred_path, 'r') as f:
            # Predicted masks: 0 (background), 1 (class 1), 2 (class 2)
            long_pred_mask = f['long_mask'][:].astype(np.uint8)
            trans_pred_mask = f['trans_mask'][:].astype(np.uint8)
        
        # Load ground truth classification label
        cls_label = -1
        if has_label:
            label_path = os.path.join(self.labels_dir, f'{case_id:04d}_label.h5')
            with h5py.File(label_path, 'r') as f:
                cls_label = int(f['cls'][()])
        
        # Load images
        image_path = os.path.join(self.images_dir, f'{case_id:04d}.h5')
        with h5py.File(image_path, 'r') as f:
            long_img = f['long_img'][:]
            trans_img = f['trans_img'][:]
        
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
        # Convert predicted masks: 1->128, 2->255 (to match original format)
        
        long_mask_128 = (long_pred_mask == 1).astype(np.uint8)  # Class 1
        long_mask_255 = (long_pred_mask == 2).astype(np.uint8)  # Class 2
        trans_mask_128 = (trans_pred_mask == 1).astype(np.uint8)
        trans_mask_255 = (trans_pred_mask == 2).astype(np.uint8)
        
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
        
        # Apply dilated masks to create 4 separate images BEFORE transforms
        long_img_128 = long_img * long_mask_128_3ch
        long_img_255 = long_img * long_mask_255_3ch
        trans_img_128 = trans_img * trans_mask_128_3ch
        trans_img_255 = trans_img * trans_mask_255_3ch
        
        # Apply transforms to each of the 4 parts
        if self.transforms:
            long_128_transformed = self.transforms(image=long_img_128)
            long_255_transformed = self.transforms(image=long_img_255)
            trans_128_transformed = self.transforms(image=trans_img_128)
            trans_255_transformed = self.transforms(image=trans_img_255)
            
            long_img_128 = long_128_transformed['image']
            long_img_255 = long_255_transformed['image']
            trans_img_128 = trans_128_transformed['image']
            trans_img_255 = trans_255_transformed['image']
        else:
            # Default: convert to tensor
            long_img_128 = torch.from_numpy(long_img_128.transpose(2, 0, 1)).float() / 255.0
            long_img_255 = torch.from_numpy(long_img_255.transpose(2, 0, 1)).float() / 255.0
            trans_img_128 = torch.from_numpy(trans_img_128.transpose(2, 0, 1)).float() / 255.0
            trans_img_255 = torch.from_numpy(trans_img_255.transpose(2, 0, 1)).float() / 255.0
        
        return {
            'long_img_128': long_img_128,  # [3, H, W]
            'long_img_255': long_img_255,  # [3, H, W]
            'trans_img_128': trans_img_128,  # [3, H, W]
            'trans_img_255': trans_img_255,  # [3, H, W]
            'cls_label': torch.tensor(cls_label).long(),
            'case_id': case_id,
            'has_label': has_label
        }


def get_csv_cls_from_seg_transforms(is_train: bool = True, image_size: int = 512):
    """
    Get albumentations transforms
    """
    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.2),
            A.GaussNoise(p=0.2),
            A.GaussianBlur(p=0.2, blur_limit=(3, 7)),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.15, 
                rotate_limit=15, 
                border_mode=0, 
                p=0.5
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.pytorch.ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.pytorch.ToTensorV2(),
        ])
