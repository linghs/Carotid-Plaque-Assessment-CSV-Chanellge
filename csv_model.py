import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Dict, Tuple


class CSVModel(nn.Module):
    """
    Multi-task model for CSV 2026 Challenge
    
    Tasks:
    1. Segmentation: Plaque segmentation (3 classes: 0=background, 1=class1, 2=class2)
    2. Classification: Risk assessment (2 classes: 0=low risk, 1=high risk)
    
    Architecture:
    - Shared encoder (EfficientNet-B4 or other backbones)
    - Segmentation head: FPN decoder + segmentation head
    - Classification head: Global pooling + FC layers
    """
    
    def __init__(
        self,
        encoder_name: str = 'efficientnet-b4',
        encoder_weights: str = 'imagenet',
        num_seg_classes: int = 2,
        num_cls_classes: int = 2,
        use_classification: bool = True
    ):
        super().__init__()
        
        self.encoder_name = encoder_name
        self.num_seg_classes = num_seg_classes
        self.num_cls_classes = num_cls_classes
        self.use_classification = use_classification
        
        # Create FPN model for segmentation
        self.fpn_model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_seg_classes,
            activation=None  # We'll apply softmax/sigmoid later
        )
        
        # Extract encoder for classification
        self.encoder = self.fpn_model.encoder
        
        # Classification head (if enabled)
        if self.use_classification:
            encoder_out_channels = self.encoder.out_channels[-1]
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(encoder_out_channels, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_cls_classes)
            )
        
        print(f"CSVModel initialized:")
        print(f"  - Encoder: {encoder_name}")
        print(f"  - Segmentation classes: {num_seg_classes}")
        print(f"  - Classification classes: {num_cls_classes}")
        print(f"  - Classification enabled: {use_classification}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            Dictionary with:
            - 'seg_logits': Segmentation logits [B, num_seg_classes, H, W]
            - 'cls_logits': Classification logits [B, num_cls_classes] (if enabled)
        """
        # Segmentation
        seg_logits = self.fpn_model(x)
        outputs = {'seg_logits': seg_logits}
        
        # Classification (using encoder features)
        if self.use_classification:
            features = self.encoder(x)
            cls_logits = self.classification_head(features[-1])
            outputs['cls_logits'] = cls_logits
        
        return outputs


class CSVLoss(nn.Module):
    """
    Combined loss for multi-task learning
    
    Loss = seg_weight * seg_loss + cls_weight * cls_loss
    """
    
    def __init__(
        self,
        seg_weight: float = 1.0,
        cls_weight: float = 1.0,
        use_dice_loss: bool = True,
        use_ce_loss: bool = True,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        ignore_index: int = -1
    ):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.use_dice_loss = use_dice_loss
        self.use_ce_loss = use_ce_loss
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index
        
        # Segmentation losses
        if use_dice_loss:
            self.dice_loss = smp.losses.DiceLoss(mode='multiclass', ignore_index=ignore_index)
        if use_ce_loss:
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        # Classification loss
        self.cls_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss
        
        Args:
            outputs: Model outputs dict with 'seg_logits' and optionally 'cls_logits'
            targets: Targets dict with 'masks' and optionally 'cls_labels'
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses for logging
        """
        loss_dict = {}
        total_loss = 0.0
        
        # Segmentation loss
        seg_logits = outputs['seg_logits']
        masks = targets['masks']
        seg_loss = 0.0
        if self.use_dice_loss:
            dice = self.dice_loss(seg_logits, masks)
            seg_loss += self.dice_weight * dice
            loss_dict['dice_loss'] = dice.item()
        
        if self.use_ce_loss:
            ce = self.ce_loss(seg_logits, masks)
            seg_loss += self.ce_weight * ce
            loss_dict['ce_loss'] = ce.item()
        
        loss_dict['seg_loss'] = seg_loss.item()
        total_loss += self.seg_weight * seg_loss
        
        # Classification loss (if available)
        if 'cls_logits' in outputs and 'cls_labels' in targets:
            cls_logits = outputs['cls_logits']
            cls_labels = targets['cls_labels']
            
            # Filter out samples without labels (cls_label == -1)
            valid_mask = (cls_labels != self.ignore_index)
            if valid_mask.any():
                cls_loss = self.cls_loss(cls_logits[valid_mask], cls_labels[valid_mask])
                loss_dict['cls_loss'] = cls_loss.item()
                total_loss += self.cls_weight * cls_loss
            else:
                loss_dict['cls_loss'] = 0.0
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict

