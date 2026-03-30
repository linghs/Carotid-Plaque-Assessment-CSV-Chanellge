import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional
import albumentations as A


class CSVDatasetSemiSupervised(Dataset):
    """
    半监督数据集：支持同时使用真实标注和伪标签数据
    
    Dataset structure:
    - Images: 1000 cases (0000.h5 to 0999.h5)
      Each h5 contains: 'long_img' and 'trans_img' (512x512 grayscale)
    - Real Labels: 200 cases (0000_label.h5 to 0199_label.h5) in labels/
    - Pseudo Labels: 800 cases (0200_label.h5 to 0999_label.h5) in pseudo_labels/
      Each h5 contains: 'long_mask', 'trans_mask' (512x512), 'cls' (0 or 1)
    
    Args:
        data_root: Path to data directory containing 'images/' and 'labels/'
        split: 'train' or 'val'
        view: 'long' or 'trans' or 'both'
        transforms: Albumentations transforms
        train_indices: List of indices for training split (from real labels 0-199)
        val_indices: List of indices for validation split (from real labels 0-199)
        use_pseudo_labels: If True, include pseudo-labeled data (cases 200-999) in training
        pseudo_labels_dir: Path to pseudo labels directory (default: data_root/pseudo_labels)
    """
    
    def __init__(
        self, 
        data_root: str,
        split: str = 'train',
        view: str = 'both',  # 'long', 'trans', or 'both'
        transforms: Optional[A.Compose] = None,
        train_indices: Optional[list] = None,
        val_indices: Optional[list] = None,
        use_pseudo_labels: bool = False,
        pseudo_labels_dir: Optional[str] = None
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.view = view
        self.transforms = transforms
        self.use_pseudo_labels = use_pseudo_labels
        
        # 图像目录
        self.images_dir = os.path.join(data_root, 'images')
        
        # 真实标签目录
        self.labels_dir = os.path.join(data_root, 'labels')
        
        # 伪标签目录
        if pseudo_labels_dir is None:
            self.pseudo_labels_dir = os.path.join(data_root, 'pseudo_labels')
        else:
            self.pseudo_labels_dir = pseudo_labels_dir
        
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        # Build dataset samples
        self.samples = []
        
        # Split labeled data into train/val (保持与原始脚本相同的划分)
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
        
        # Add real labeled samples (0-199)
        for idx in indices:
            if view in ['long', 'both']:
                self.samples.append({
                    'case_id': idx,
                    'view': 'long',
                    'has_label': True,
                    'is_pseudo': False
                })
            if view in ['trans', 'both']:
                self.samples.append({
                    'case_id': idx,
                    'view': 'trans',
                    'has_label': True,
                    'is_pseudo': False
                })
        
        # Add pseudo-labeled samples for training (200-999)
        if split == 'train' and use_pseudo_labels:
            if not os.path.exists(self.pseudo_labels_dir):
                print(f"Warning: Pseudo labels directory not found: {self.pseudo_labels_dir}")
                print("Training will only use real labeled data.")
            else:
                # 查找所有伪标签文件
                pseudo_label_files = [f for f in os.listdir(self.pseudo_labels_dir) 
                                     if f.endswith('_label.h5')]
                pseudo_indices = []
                for filename in pseudo_label_files:
                    try:
                        case_id = int(filename.split('_')[0])
                        if case_id >= 200:  # 只使用200-999的伪标签
                            pseudo_indices.append(case_id)
                    except:
                        continue
                
                pseudo_indices = sorted(pseudo_indices)
                
                for idx in pseudo_indices:
                    if view in ['long', 'both']:
                        self.samples.append({
                            'case_id': idx,
                            'view': 'long',
                            'has_label': True,
                            'is_pseudo': True
                        })
                    if view in ['trans', 'both']:
                        self.samples.append({
                            'case_id': idx,
                            'view': 'trans',
                            'has_label': True,
                            'is_pseudo': True
                        })
                
                print(f"Added {len(pseudo_indices)} pseudo-labeled cases to training set")
        
        # 统计信息
        num_real = sum(1 for s in self.samples if not s['is_pseudo'])
        num_pseudo = sum(1 for s in self.samples if s['is_pseudo'])
        
        print(f"CSV Dataset [{split}] initialized:")
        print(f"  - View: {view}")
        print(f"  - Real labeled: {num_real} samples")
        if use_pseudo_labels and split == 'train':
            print(f"  - Pseudo labeled: {num_pseudo} samples")
        print(f"  - Total: {len(self.samples)} samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sample_info = self.samples[idx]
        case_id = sample_info['case_id']
        view_type = sample_info['view']
        has_label = sample_info['has_label']
        is_pseudo = sample_info['is_pseudo']
        
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
            # 根据是否为伪标签选择不同的目录
            if is_pseudo:
                label_path = os.path.join(self.pseudo_labels_dir, f'{case_id:04d}_label.h5')
            else:
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
            'has_label': has_label,
            'is_pseudo': is_pseudo
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
