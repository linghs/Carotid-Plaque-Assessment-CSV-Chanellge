import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional
import albumentations as A


class CSVDataset(Dataset):
    """
    Dataset for CSV 2026 Challenge: Carotid Plaque Segmentation and Vulnerability Assessment
    
    Dataset structure:
    - Images: 1000 cases (0000.h5 to 0999.h5)
      Each h5 contains: 'long_img' and 'trans_img' (512x512 grayscale)
    - Labels: 200 cases (0000_label.h5 to 0199_label.h5)
      Each h5 contains: 'long_mask', 'trans_mask' (512x512), 'cls' (0 or 1)
    
    Args:
        data_root: Path to data directory containing 'images/' and 'labels/'
        split: 'train' or 'val'
        view: 'long' or 'trans' or 'both'
        transforms: Albumentations transforms
        train_indices: List of indices for training split
        val_indices: List of indices for validation split
        use_unlabeled: If True, include unlabeled data (cases 200-999) for semi-supervised learning
    """
    
    def __init__(
        self, 
        data_root: str,
        split: str = 'train',
        view: str = 'both',  # 'long', 'trans', or 'both'
        transforms: Optional[A.Compose] = None,
        train_indices: Optional[list] = None,
        val_indices: Optional[list] = None,
        use_unlabeled: bool = False
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.view = view
        self.transforms = transforms
        self.use_unlabeled = use_unlabeled
        
        self.images_dir = os.path.join(data_root, 'images')
        self.labels_dir = os.path.join(data_root, 'labels')
        
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        # Build dataset samples
        self.samples = []
        
        # Labeled data (0-199)
        labeled_indices = list(range(200))
        
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
        
        # Add labeled samples
        for idx in indices:
            if view in ['long', 'both']:
                self.samples.append({
                    'case_id': idx,
                    'view': 'long',
                    'has_label': True
                })
            if view in ['trans', 'both']:
                self.samples.append({
                    'case_id': idx,
                    'view': 'trans',
                    'has_label': True
                })
        
        # Add unlabeled samples for training (semi-supervised)
        if split == 'train' and use_unlabeled:
            unlabeled_indices = list(range(200, 1000))
            for idx in unlabeled_indices:
                if view in ['long', 'both']:
                    self.samples.append({
                        'case_id': idx,
                        'view': 'long',
                        'has_label': False
                    })
                if view in ['trans', 'both']:
                    self.samples.append({
                        'case_id': idx,
                        'view': 'trans',
                        'has_label': False
                    })
        
        print(f"CSV Dataset [{split}] initialized: {len(self.samples)} samples (view: {view}, use_unlabeled: {use_unlabeled})")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sample_info = self.samples[idx]
        case_id = sample_info['case_id']
        view_type = sample_info['view']
        has_label = sample_info['has_label']
        
        # Load image
        image_path = os.path.join(self.images_dir, f'{case_id:04d}.h5')
        with h5py.File(image_path, 'r') as f:
            if view_type == 'long':
                image = f['long_img'][:]
            else:  # trans
                image = f['trans_img'][:]
        
        # Convert to 3-channel (grayscale to RGB)
        image = np.stack([image, image, image], axis=-1).astype(np.float32)
        
        # Normalize to [0, 255] if needed
        if image.max() <= 1.0:
            image = image * 255.0
        image = image.astype(np.uint8)
        
        # Load mask and classification label if available
        mask = None
        cls_label = -1  # -1 indicates no label
        
        if has_label:
            label_path = os.path.join(self.labels_dir, f'{case_id:04d}_label.h5')
            with h5py.File(label_path, 'r') as f:
                if view_type == 'long':
                    mask = f['long_mask'][:]
                else:  # trans
                    mask = f['trans_mask'][:]
                cls_label = int(f['cls'][()])
            
            # Convert mask to 3-class format
            # Original: 0 (background), 128 (class 1), 255 (class 2)
            # Convert to: 0 (background), 1 (class 1), 2 (class 2)
            mask = mask.astype(np.float32)
            mask[mask == 128] = 1
            mask[mask == 255] = 2
            mask = mask.astype(np.uint8)
        else:
            # Create dummy mask for unlabeled data
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # Default: convert to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()
        
        # Ensure mask is long type
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        elif mask.dtype != torch.long:
            mask = mask.long()
        
        return {
            'image': image,
            'mask': mask,
            'cls_label': torch.tensor(cls_label).long(),
            'case_id': case_id,
            'view': view_type,
            'has_label': has_label
        }


def get_csv_transforms(is_train: bool = True, image_size: int = 512):
    """
    Get albumentations transforms for CSV dataset
    
    Args:
        is_train: If True, apply augmentation
        image_size: Target image size (default 512)
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

