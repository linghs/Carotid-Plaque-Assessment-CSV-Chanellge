import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import torch.nn.functional as F
from typing import Dict, List, Tuple
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt


def calculate_dice_coefficient(pred_mask: torch.Tensor, true_mask: torch.Tensor, num_classes: int = 2) -> Dict[str, float]:
    """
    Calculate Dice coefficient for each class
    
    Args:
        pred_mask: Predicted segmentation [B, H, W]
        true_mask: Ground truth segmentation [B, H, W]
        num_classes: Number of classes
    
    Returns:
        Dictionary with dice scores
    """
    dice_scores = {}
    
    # Clamp predictions and targets to valid range [0, num_classes-1]
    pred_mask = torch.clamp(pred_mask, 0, num_classes - 1)
    true_mask = torch.clamp(true_mask, 0, num_classes - 1)
    
    # Convert to one-hot
    pred_one_hot = F.one_hot(pred_mask, num_classes=num_classes).permute(0, 3, 1, 2).float()
    true_one_hot = F.one_hot(true_mask, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Calculate dice for each class (excluding background)
    for c in range(1, num_classes):
        pred_c = pred_one_hot[:, c]
        true_c = true_one_hot[:, c]
        
        intersection = torch.sum(pred_c * true_c)
        union = torch.sum(pred_c) + torch.sum(true_c)
        
        if union > 0:
            dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
        else:
            dice = torch.tensor(1.0)  # Perfect score if both are empty
        
        dice_scores[f'dice_class_{c}'] = dice.item()
    
    # Mean dice (excluding background)
    if len(dice_scores) > 0:
        dice_scores['dice_mean'] = np.mean(list(dice_scores.values()))
    
    return dice_scores


def calculate_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor, num_classes: int = 2) -> Dict[str, float]:
    """
    Calculate IoU (Intersection over Union) for each class
    
    Args:
        pred_mask: Predicted segmentation [B, H, W]
        true_mask: Ground truth segmentation [B, H, W]
        num_classes: Number of classes
    
    Returns:
        Dictionary with IoU scores
    """
    iou_scores = {}
    
    # Clamp predictions and targets to valid range [0, num_classes-1]
    pred_mask = torch.clamp(pred_mask, 0, num_classes - 1)
    true_mask = torch.clamp(true_mask, 0, num_classes - 1)
    
    # Convert to one-hot
    pred_one_hot = F.one_hot(pred_mask, num_classes=num_classes).permute(0, 3, 1, 2).float()
    true_one_hot = F.one_hot(true_mask, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Calculate IoU for each class (excluding background)
    for c in range(1, num_classes):
        pred_c = pred_one_hot[:, c]
        true_c = true_one_hot[:, c]
        
        intersection = torch.sum(pred_c * true_c)
        union = torch.sum(pred_c) + torch.sum(true_c) - intersection
        
        if union > 0:
            iou = (intersection + 1e-7) / (union + 1e-7)
        else:
            iou = torch.tensor(1.0)  # Perfect score if both are empty
        
        iou_scores[f'iou_class_{c}'] = iou.item()
    
    # Mean IoU (excluding background)
    if len(iou_scores) > 0:
        iou_scores['iou_mean'] = np.mean(list(iou_scores.values()))
    
    return iou_scores


def calculate_classification_metrics(pred_logits: torch.Tensor, true_labels: torch.Tensor) -> Dict[str, float]:
    """
    Calculate classification metrics
    
    Args:
        pred_logits: Predicted logits [B, num_classes]
        true_labels: Ground truth labels [B]
    
    Returns:
        Dictionary with metrics
    """
    # Get predictions
    pred_probs = F.softmax(pred_logits, dim=1)
    pred_labels = torch.argmax(pred_logits, dim=1)
    
    # Convert to numpy (detach first to avoid grad issues)
    pred_labels_np = pred_labels.detach().cpu().numpy()
    true_labels_np = true_labels.detach().cpu().numpy()
    pred_probs_np = pred_probs.detach().cpu().numpy()
    
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(true_labels_np, pred_labels_np)
    
    # F1 Score
    metrics['f1_macro'] = f1_score(true_labels_np, pred_labels_np, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(true_labels_np, pred_labels_np, average='weighted', zero_division=0)
    
    # AUC (for binary and multi-class)
    try:
        if pred_probs_np.shape[1] == 2:
            # Binary classification
            metrics['auc'] = roc_auc_score(true_labels_np, pred_probs_np[:, 1])
        else:
            # Multi-class
            metrics['auc'] = roc_auc_score(true_labels_np, pred_probs_np, multi_class='ovr', average='macro')
    except:
        metrics['auc'] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(true_labels_np, pred_labels_np)
    
    # Class-wise metrics
    for i in range(pred_probs_np.shape[1]):
        if i < len(cm):
            # True positives, false positives, false negatives
            tp = cm[i, i] if i < len(cm) and i < cm.shape[1] else 0
            fp = cm[:, i].sum() - tp if i < cm.shape[1] else 0
            fn = cm[i, :].sum() - tp if i < len(cm) else 0
            
            # Precision, Recall
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            
            metrics[f'precision_class_{i}'] = precision
            metrics[f'recall_class_{i}'] = recall
    
    return metrics


def calculate_normalized_surface_distance(pred_mask: np.ndarray, true_mask: np.ndarray, 
                                          spacing: Tuple[float, float] = (1.0, 1.0),
                                          tolerance: float = 2.0) -> float:
    """
    Calculate Normalized Surface Distance (NSD)
    
    Args:
        pred_mask: Predicted binary mask [H, W]
        true_mask: Ground truth binary mask [H, W]
        spacing: Pixel spacing (default: 1.0, 1.0)
        tolerance: Distance tolerance in mm (default: 2.0)
    
    Returns:
        NSD score (0-1, higher is better)
    """
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)
    
    # If both masks are empty, return perfect score
    if not pred_mask.any() and not true_mask.any():
        return 1.0
    
    # If one is empty and the other isn't, return 0
    if not pred_mask.any() or not true_mask.any():
        return 0.0
    
    # Get boundaries
    pred_border = pred_mask ^ np.roll(pred_mask, 1, axis=0) | pred_mask ^ np.roll(pred_mask, 1, axis=1)
    true_border = true_mask ^ np.roll(true_mask, 1, axis=0) | true_mask ^ np.roll(true_mask, 1, axis=1)
    
    # If no borders found, fall back to simple comparison
    if not pred_border.any() or not true_border.any():
        return float(np.all(pred_mask == true_mask))
    
    # Calculate distance transforms
    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)
    dt_true = distance_transform_edt(~true_border, sampling=spacing)
    
    # Get surface points
    pred_surface_pts = np.argwhere(pred_border)
    true_surface_pts = np.argwhere(true_border)
    
    # Calculate distances from predicted surface to true mask
    pred_distances = dt_true[pred_border]
    true_distances = dt_pred[true_border]
    
    # Count points within tolerance
    pred_within_tolerance = np.sum(pred_distances <= tolerance)
    true_within_tolerance = np.sum(true_distances <= tolerance)
    
    # Calculate NSD
    total_surface_points = len(pred_distances) + len(true_distances)
    if total_surface_points == 0:
        return 1.0
    
    nsd = (pred_within_tolerance + true_within_tolerance) / total_surface_points
    
    return nsd


def calculate_dsc_nsd_for_multiclass(pred_mask: torch.Tensor, true_mask: torch.Tensor,
                                      class_value: int, tolerance: float = 2.0) -> Tuple[float, float]:
    """
    Calculate DSC and NSD for a specific class
    
    Args:
        pred_mask: Predicted segmentation [H, W]
        true_mask: Ground truth segmentation [H, W]
        class_value: Class value to evaluate (128 for plaque, 255 for vessel)
        tolerance: Distance tolerance for NSD
    
    Returns:
        Tuple of (DSC, NSD)
    """
    # Extract binary masks for the class
    pred_binary = (pred_mask == class_value).float()
    true_binary = (true_mask == class_value).float()
    
    # Calculate DSC
    intersection = torch.sum(pred_binary * true_binary)
    union = torch.sum(pred_binary) + torch.sum(true_binary)
    
    if union > 0:
        dsc = (2.0 * intersection / (union + 1e-7)).item()
    else:
        dsc = 1.0 if intersection == 0 else 0.0
    
    # Calculate NSD (convert to numpy)
    pred_np = pred_binary.cpu().numpy()
    true_np = true_binary.cpu().numpy()
    nsd = calculate_normalized_surface_distance(pred_np, true_np, tolerance=tolerance)
    
    return dsc, nsd


class MetricsTracker:
    """
    Track and aggregate metrics following the paper's evaluation protocol:
    - Segmentation Score (S_seg): Based on longitudinal and transversal view scores
    - Classification Score (S_cls): Average F1-score across all classes
    
    Segmentation classes:
        0: Background
        128: Plaque
        255: Vessel
    """
    
    def __init__(self, num_cls_classes: int = 2, tolerance: float = 2.0):
        """
        Args:
            num_cls_classes: Number of classification classes
            tolerance: Distance tolerance for NSD calculation (in mm)
        """
        self.num_cls_classes = num_cls_classes
        self.tolerance = tolerance
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics"""
        # Segmentation metrics by view type
        self.long_view_metrics = []
        self.trans_view_metrics = []
        
        # Classification metrics
        self.cls_predictions = []
        self.cls_labels = []
        
        # Loss tracking
        self.losses = []
    
    def update(
        self, 
        pred_seg: torch.Tensor,
        true_seg: torch.Tensor,
        view_type: str = 'long',
        pred_cls_logits: torch.Tensor = None,
        true_cls_labels: torch.Tensor = None,
        loss_dict: Dict[str, float] = None
    ):
        """
        Update metrics with batch results
        
        Args:
            pred_seg: Predicted segmentation [B, H, W] with values {0, 128, 255}
            true_seg: Ground truth segmentation [B, H, W] with values {0, 128, 255}
            view_type: 'long' for longitudinal or 'trans' for transversal view
            pred_cls_logits: Predicted classification logits [B, num_classes]
            true_cls_labels: Ground truth classification labels [B]
            loss_dict: Dictionary of losses
        """
        batch_size = pred_seg.shape[0]
        
        # Ensure true_seg is in {0, 128, 255} format
        # If it's in {0, 1, 2} format, convert it
        unique_vals = torch.unique(true_seg)
        if unique_vals.max() <= 2:
            # Convert from {0, 1, 2} to {0, 128, 255}
            true_seg_converted = torch.zeros_like(true_seg)
            true_seg_converted[true_seg == 1] = 128  # Plaque
            true_seg_converted[true_seg == 2] = 255  # Vessel
            true_seg = true_seg_converted
        
        # Calculate segmentation metrics for each sample in batch
        for i in range(batch_size):
            pred_i = pred_seg[i]
            true_i = true_seg[i]
            
            # Calculate metrics for plaque (128)
            dsc_plaque, nsd_plaque = calculate_dsc_nsd_for_multiclass(
                pred_i, true_i, class_value=128, tolerance=self.tolerance
            )
            
            # Calculate metrics for vessel (255)
            dsc_vessel, nsd_vessel = calculate_dsc_nsd_for_multiclass(
                pred_i, true_i, class_value=255, tolerance=self.tolerance
            )
            
            # Calculate S_v,plaque = (DSC_plaque + NSD_plaque) / 2
            s_v_plaque = (dsc_plaque + nsd_plaque) / 2.0
            
            # Calculate S_v,vessel = (DSC_vessel + NSD_vessel) / 2
            s_v_vessel = (dsc_vessel + nsd_vessel) / 2.0
            
            # Calculate S_v = 0.6 * S_v,plaque + 0.4 * S_v,vessel
            s_v = 0.6 * s_v_plaque + 0.4 * s_v_vessel
            
            # Store metrics
            metrics = {
                'dsc_plaque': dsc_plaque,
                'nsd_plaque': nsd_plaque,
                's_v_plaque': s_v_plaque,
                'dsc_vessel': dsc_vessel,
                'nsd_vessel': nsd_vessel,
                's_v_vessel': s_v_vessel,
                's_v': s_v
            }
            
            # Add to appropriate view list
            if view_type == 'long':
                self.long_view_metrics.append(metrics)
            else:
                self.trans_view_metrics.append(metrics)
        
        # Store classification predictions
        if pred_cls_logits is not None and true_cls_labels is not None:
            pred_labels = torch.argmax(pred_cls_logits, dim=1)
            self.cls_predictions.extend(pred_labels.cpu().numpy().tolist())
            self.cls_labels.extend(true_cls_labels.cpu().numpy().tolist())
        
        # Loss tracking
        if loss_dict is not None:
            self.losses.append(loss_dict)
    
    def get_segmentation_score(self) -> Dict[str, float]:
        """
        Calculate Segmentation Score (S_seg) following the paper's formula:
        S_seg = 0.5 * S_long + 0.5 * S_trans
        where S_v = 0.6 * S_v,plaque + 0.4 * S_v,vessel
        
        Returns:
            Dictionary with all segmentation metrics
        """
        result = {}
        
        # Calculate average metrics for longitudinal view
        if self.long_view_metrics:
            for key in self.long_view_metrics[0].keys():
                values = [m[key] for m in self.long_view_metrics]
                result[f'long_{key}'] = np.mean(values)
            result['s_long'] = result['long_s_v']
        else:
            result['s_long'] = 0.0
        
        # Calculate average metrics for transversal view
        if self.trans_view_metrics:
            for key in self.trans_view_metrics[0].keys():
                values = [m[key] for m in self.trans_view_metrics]
                result[f'trans_{key}'] = np.mean(values)
            result['s_trans'] = result['trans_s_v']
        else:
            result['s_trans'] = 0.0
        
        # Calculate final Segmentation Score
        result['segmentation_score'] = 0.5 * result['s_long'] + 0.5 * result['s_trans']
        
        return result
    
    def get_classification_score(self) -> Dict[str, float]:
        """
        Calculate Classification Score (S_cls) following the paper's formula:
        S_cls = (1/N) * Σ F1_i
        where N is the number of classes
        
        Returns:
            Dictionary with classification metrics including per-class F1 scores
        """
        result = {}
        
        if not self.cls_predictions or not self.cls_labels:
            result['classification_score'] = 0.0
            return result
        
        cls_pred = np.array(self.cls_predictions)
        cls_true = np.array(self.cls_labels)
        
        # Filter out invalid labels (-1)
        valid_mask = cls_true != -1
        if not valid_mask.any():
            result['classification_score'] = 0.0
            return result
        
        cls_pred = cls_pred[valid_mask]
        cls_true = cls_true[valid_mask]
        
        # Calculate per-class F1 scores
        f1_per_class = []
        for i in range(self.num_cls_classes):
            # Binary mask for current class
            pred_i = (cls_pred == i)
            true_i = (cls_true == i)
            
            # Calculate F1 for this class
            tp = np.sum(pred_i & true_i)
            fp = np.sum(pred_i & ~true_i)
            fn = np.sum(~pred_i & true_i)
            
            if tp + fp + fn == 0:
                f1_i = 1.0
            else:
                precision = tp / (tp + fp + 1e-7)
                recall = tp / (tp + fn + 1e-7)
                f1_i = 2 * precision * recall / (precision + recall + 1e-7)
            
            f1_per_class.append(f1_i)
            result[f'f1_class_{i}'] = f1_i
        
        # Calculate Classification Score (average F1)
        result['classification_score'] = np.mean(f1_per_class)
        
        # Also calculate overall metrics
        result['accuracy'] = accuracy_score(cls_true, cls_pred)
        result['f1_macro'] = f1_score(cls_true, cls_pred, average='macro', zero_division=0)
        
        return result
    
    def get_all_metrics(self) -> Dict[str, float]:
        """
        Get all metrics including segmentation score, classification score, and losses
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Segmentation metrics
        seg_metrics = self.get_segmentation_score()
        metrics.update(seg_metrics)
        
        # Classification metrics
        cls_metrics = self.get_classification_score()
        metrics.update(cls_metrics)
        
        # Loss metrics
        if self.losses:
            for key in self.losses[0].keys():
                values = [m[key] for m in self.losses]
                metrics[f'avg_{key}'] = np.mean(values)
        
        return metrics
    
    def print_metrics(self, prefix: str = ""):
        """Print metrics in a formatted way following the paper's structure"""
        metrics = self.get_all_metrics()
        
        print(f"\n{'='*80}")
        print(f"{prefix} Evaluation Metrics")
        print(f"{'='*80}")
        
        # Segmentation Score
        print(f"\n📊 SEGMENTATION SCORE: {metrics.get('segmentation_score', 0.0):.4f}")
        print(f"{'─'*80}")
        
        # Longitudinal view
        print(f"\n  Longitudinal View (S_long = {metrics.get('s_long', 0.0):.4f}):")
        print(f"    Plaque (weight=0.6):")
        print(f"      DSC_plaque:     {metrics.get('long_dsc_plaque', 0.0):.4f}")
        print(f"      NSD_plaque:     {metrics.get('long_nsd_plaque', 0.0):.4f}")
        print(f"      S_v,plaque:     {metrics.get('long_s_v_plaque', 0.0):.4f}")
        print(f"    Vessel (weight=0.4):")
        print(f"      DSC_vessel:     {metrics.get('long_dsc_vessel', 0.0):.4f}")
        print(f"      NSD_vessel:     {metrics.get('long_nsd_vessel', 0.0):.4f}")
        print(f"      S_v,vessel:     {metrics.get('long_s_v_vessel', 0.0):.4f}")
        
        # Transversal view
        print(f"\n  Transversal View (S_trans = {metrics.get('s_trans', 0.0):.4f}):")
        print(f"    Plaque (weight=0.6):")
        print(f"      DSC_plaque:     {metrics.get('trans_dsc_plaque', 0.0):.4f}")
        print(f"      NSD_plaque:     {metrics.get('trans_nsd_plaque', 0.0):.4f}")
        print(f"      S_v,plaque:     {metrics.get('trans_s_v_plaque', 0.0):.4f}")
        print(f"    Vessel (weight=0.4):")
        print(f"      DSC_vessel:     {metrics.get('trans_dsc_vessel', 0.0):.4f}")
        print(f"      NSD_vessel:     {metrics.get('trans_nsd_vessel', 0.0):.4f}")
        print(f"      S_v,vessel:     {metrics.get('trans_s_v_vessel', 0.0):.4f}")
        
        # Classification Score
        print(f"\n{'─'*80}")
        print(f"\n🎯 CLASSIFICATION SCORE: {metrics.get('classification_score', 0.0):.4f}")
        print(f"{'─'*80}")
        if metrics.get('classification_score', 0.0) > 0:
            for i in range(self.num_cls_classes):
                f1_key = f'f1_class_{i}'
                if f1_key in metrics:
                    print(f"  F1 Class {i}:       {metrics[f1_key]:.4f}")
            print(f"  Accuracy:         {metrics.get('accuracy', 0.0):.4f}")
            print(f"  F1 Macro:         {metrics.get('f1_macro', 0.0):.4f}")
        else:
            print("  No classification data available")
        
        # Losses
        if any('loss' in k for k in metrics.keys()):
            print(f"\n{'─'*80}")
            print(f"\n💹 Losses:")
            for key, value in metrics.items():
                if 'loss' in key:
                    print(f"  {key:20s}: {value:.4f}")
        
        print(f"\n{'='*80}\n")

