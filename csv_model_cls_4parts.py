import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Dict


class CSVClassificationModel4Parts(nn.Module):
    """
    4-Part Classification model using 4 separate encoders
    
    Architecture:
    - 4 independent encoders (one for each masked image part):
      1. long_img_128 (longitudinal view with mask==128)
      2. long_img_255 (longitudinal view with mask==255)
      3. trans_img_128 (transverse view with mask==128)
      4. trans_img_255 (transverse view with mask==255)
    - Feature fusion (concat / add / attention)
    - Classification head
    
    Args:
        encoder_name: Encoder backbone name (e.g., 'resnet152', 'efficientnet-b4')
        encoder_weights: Pretrained weights ('imagenet' or None)
        num_cls_classes: Number of classification classes (default: 2)
        fusion_method: Feature fusion method ('concat', 'add', 'attention')
    """
    
    def __init__(
        self,
        encoder_name: str = 'efficientnet-b4',
        encoder_weights: str = 'imagenet',
        num_cls_classes: int = 2,
        fusion_method: str = 'concat'
    ):
        super().__init__()
        
        self.encoder_name = encoder_name
        self.num_cls_classes = num_cls_classes
        self.fusion_method = fusion_method
        
        # Create 4 separate encoders (one for each image part)
        self.long_128_encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights
        )
        
        self.long_255_encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights
        )
        
        self.trans_128_encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights
        )
        
        self.trans_255_encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights
        )
        
        # Get encoder output channels
        encoder_out_channels = self.long_128_encoder.out_channels[-1]
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature fusion
        if fusion_method == 'concat':
            # Concatenate all 4 encoder features
            feature_dim = encoder_out_channels * 4
        elif fusion_method == 'add':
            # Element-wise addition of all 4 features
            feature_dim = encoder_out_channels
        elif fusion_method == 'attention':
            # Attention-based fusion
            feature_dim = encoder_out_channels
            self.attention = nn.Sequential(
                nn.Linear(encoder_out_channels * 4, encoder_out_channels),
                nn.Tanh(),
                nn.Linear(encoder_out_channels, 4),
                nn.Softmax(dim=1)
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
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
        
        print(f"CSVClassificationModel4Parts initialized:")
        print(f"  - Encoder: {encoder_name}")
        print(f"  - Encoder output channels: {encoder_out_channels}")
        print(f"  - Fusion method: {fusion_method}")
        print(f"  - Feature dim: {feature_dim}")
        print(f"  - Classification classes: {num_cls_classes}")
    
    def forward(
        self, 
        long_img_128: torch.Tensor,
        long_img_255: torch.Tensor,
        trans_img_128: torch.Tensor,
        trans_img_255: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with 4 separate image inputs
        
        Args:
            long_img_128: Longitudinal image with mask==128 [B, 3, H, W]
            long_img_255: Longitudinal image with mask==255 [B, 3, H, W]
            trans_img_128: Transverse image with mask==128 [B, 3, H, W]
            trans_img_255: Transverse image with mask==255 [B, 3, H, W]
        
        Returns:
            Dictionary with 'cls_logits': [B, num_cls_classes]
        """
        # Extract features from each of the 4 parts
        feat_long_128 = self.long_128_encoder(long_img_128)[-1]  # [B, C, H', W']
        feat_long_255 = self.long_255_encoder(long_img_255)[-1]  # [B, C, H', W']
        feat_trans_128 = self.trans_128_encoder(trans_img_128)[-1]  # [B, C, H', W']
        feat_trans_255 = self.trans_255_encoder(trans_img_255)[-1]  # [B, C, H', W']
        
        # Global average pooling
        feat_long_128 = self.global_pool(feat_long_128).flatten(1)  # [B, C]
        feat_long_255 = self.global_pool(feat_long_255).flatten(1)  # [B, C]
        feat_trans_128 = self.global_pool(feat_trans_128).flatten(1)  # [B, C]
        feat_trans_255 = self.global_pool(feat_trans_255).flatten(1)  # [B, C]
        
        # Feature fusion
        if self.fusion_method == 'concat':
            # Concatenate all features
            fused_features = torch.cat([
                feat_long_128, feat_long_255, 
                feat_trans_128, feat_trans_255
            ], dim=1)  # [B, 4*C]
        
        elif self.fusion_method == 'add':
            # Element-wise addition
            fused_features = feat_long_128 + feat_long_255 + feat_trans_128 + feat_trans_255  # [B, C]
        
        elif self.fusion_method == 'attention':
            # Attention-based fusion
            # First concatenate all features
            all_features = torch.cat([
                feat_long_128, feat_long_255,
                feat_trans_128, feat_trans_255
            ], dim=1)  # [B, 4*C]
            
            # Compute attention weights
            attention_weights = self.attention(all_features)  # [B, 4]
            
            # Stack features and apply attention
            features_stack = torch.stack([
                feat_long_128, feat_long_255,
                feat_trans_128, feat_trans_255
            ], dim=1)  # [B, 4, C]
            
            # Apply attention: [B, 4, 1] * [B, 4, C] -> [B, 4, C] -> sum -> [B, C]
            fused_features = (attention_weights.unsqueeze(2) * features_stack).sum(dim=1)
        
        # Classification
        cls_logits = self.classification_head(fused_features)  # [B, num_cls_classes]
        
        return {'cls_logits': cls_logits}


class CSVClassificationLoss(nn.Module):
    """
    Classification loss for CSVClassificationModel4Parts
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
            loss: Total loss
            loss_dict: Dictionary with loss components for logging
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
