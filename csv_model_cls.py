import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Dict


class CSVClassificationModel(nn.Module):
    """
    Classification model that uses mask information with optional encoder support
    
    Architecture:
    - Can use either:
      1. Mask-only mode (use_encoder=False): Simple CNN on masks
      2. Encoder mode (use_encoder=True): Image encoder + mask features
    
    The masks contain: 0=background, 128=class1, 255=class2
    """
    
    def __init__(
        self,
        encoder_name: str = 'efficientnet-b4',
        encoder_weights: str = 'imagenet',
        num_cls_classes: int = 2,
        fusion_method: str = 'concat',
        use_mask_features: bool = True,
        use_encoder: bool = False  # NEW: Control whether to use encoder
    ):
        super().__init__()
        
        self.num_cls_classes = num_cls_classes
        self.use_encoder = use_encoder
        self.fusion_method = fusion_method
        self.encoder_name = encoder_name
        
        if use_encoder:
            # Use image encoder backbone
            # Create a dummy model to get encoder
            dummy_model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=1,
            )
            self.encoder = dummy_model.encoder
            encoder_out_channels = self.encoder.out_channels[-1]
            
            # Global pooling for encoder features
            self.encoder_pool = nn.AdaptiveAvgPool2d(1)
            
            # Feature dimension from encoder (pooled from both views)
            if fusion_method == 'concat':
                feature_dim = encoder_out_channels * 2  # Both views concatenated
            else:
                feature_dim = encoder_out_channels  # Add or max fusion
            
            if use_mask_features:
                # Mask processing for additional features
                self.mask_processor = nn.Sequential(
                    nn.Conv2d(2, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten()
                )
                mask_feature_dim = 64
                
                if fusion_method == 'concat':
                    feature_dim = feature_dim + mask_feature_dim
            
            print(f"CSVClassificationModel (Encoder Mode) initialized:")
            print(f"  - Encoder: {encoder_name}")
            print(f"  - Encoder output channels: {encoder_out_channels}")
            print(f"  - Use mask features: {use_mask_features}")
            print(f"  - Fusion method: {fusion_method}")
            print(f"  - Final feature dim: {feature_dim}")
            
        else:
            # Mask-only mode (original)
            self.mask_processor = nn.Sequential(
                # First conv block
                nn.Conv2d(2, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 512 -> 256
                
                # Second conv block
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 256 -> 128
                
                # Third conv block
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 128 -> 64
                
                # Fourth conv block
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 64 -> 32
                
                # Global pooling
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            feature_dim = 256
            
            print(f"CSVClassificationModel (Mask-Only) initialized:")
            print(f"  - Classification classes: {num_cls_classes}")
            print(f"  - Input: long_mask + trans_mask (2 channels)")
            print(f"  - Mask feature dim: {feature_dim}")
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_cls_classes)
        )
        
        print(f"  - Classification classes: {num_cls_classes}")
    
    def forward(self, long_mask: torch.Tensor, trans_mask: torch.Tensor, 
                long_img: torch.Tensor = None, trans_img: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            long_mask: Longitudinal view masks [B, 1, H, W]
            trans_mask: Transverse view masks [B, 1, H, W]
            long_img: Optional longitudinal images [B, 3, H, W] (for encoder mode)
            trans_img: Optional transverse images [B, 3, H, W] (for encoder mode)
        
        Returns:
            Dictionary with:
            - 'cls_logits': Classification logits [B, num_cls_classes]
        """
        if self.use_encoder:
            # Encoder mode: use images
            if long_img is None or trans_img is None:
                raise ValueError("Images required when use_encoder=True")
            
            # Process both views through encoder
            long_features = self.encoder(long_img)[-1]  # [B, C, H', W']
            trans_features = self.encoder(trans_img)[-1]  # [B, C, H', W']
            
            # Pool encoder features first
            long_pooled = self.encoder_pool(long_features).flatten(1)  # [B, encoder_dim]
            trans_pooled = self.encoder_pool(trans_features).flatten(1)  # [B, encoder_dim]
            
            # Combine features from both views
            if self.fusion_method == 'concat':
                encoder_features = torch.cat([long_pooled, trans_pooled], dim=1)
            elif self.fusion_method == 'add':
                encoder_features = long_pooled + trans_pooled
            elif self.fusion_method == 'max':
                encoder_features = torch.max(long_pooled, trans_pooled)
            else:  # default to concat
                encoder_features = torch.cat([long_pooled, trans_pooled], dim=1)
            
            # Add mask features if enabled
            if hasattr(self, 'mask_processor') and hasattr(self.mask_processor, '__call__'):
                combined_masks = torch.cat([long_mask, trans_mask], dim=1)  # [B, 2, H, W]
                mask_features = self.mask_processor(combined_masks)  # [B, mask_dim]
                
                if self.fusion_method == 'concat':
                    features = torch.cat([encoder_features, mask_features], dim=1)
                else:
                    features = encoder_features
            else:
                features = encoder_features
        else:
            # Mask-only mode (original)
            combined_masks = torch.cat([long_mask, trans_mask], dim=1)  # [B, 2, H, W]
            features = self.mask_processor(combined_masks)  # [B, 256]
        
        # Classification
        cls_logits = self.classification_head(features)  # [B, num_cls_classes]
        
        return {'cls_logits': cls_logits}


class CSVClassificationLoss(nn.Module):
    """
    Classification loss
    """
    
    def __init__(
        self,
        class_weights: torch.Tensor = None,
        ignore_index: int = -1
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.cls_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
    
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ):
        """
        Compute classification loss
        
        Args:
            outputs: Model outputs dict with 'cls_logits'
            targets: Classification labels [B]
        
        Returns:
            loss: Classification loss
            loss_dict: Dictionary with loss for logging
        """
        cls_logits = outputs['cls_logits']
        
        # Filter out samples without labels (label == -1)
        valid_mask = (targets != self.ignore_index)
        
        if valid_mask.any():
            loss = self.cls_loss(cls_logits[valid_mask], targets[valid_mask])
        else:
            loss = torch.tensor(0.0, device=cls_logits.device)
        
        loss_dict = {
            'cls_loss': loss.item(),
            'total_loss': loss.item()
        }
        
        return loss, loss_dict

