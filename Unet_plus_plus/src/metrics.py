"""
Evaluation Metrics for Semantic Segmentation
Implements per-class IoU, Dice, F1, Precision, and Recall
"""
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


class SegmentationMetrics:
    """
    Computes segmentation metrics including IoU, Dice, F1, Precision, and Recall
    with per-class and aggregate statistics.
    """
    def __init__(self, num_classes, ignore_index=255, class_names=None):
        """
        Args:
            num_classes: Number of classes
            ignore_index: Index to ignore in calculations
            class_names: Optional list of class names for reporting
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        
    def update(self, predictions, targets):
        """
        Update metrics with new predictions and targets
        
        Args:
            predictions: (B, H, W) - predicted class labels
            targets: (B, H, W) - ground truth labels
        """
        # Flatten arrays
        pred_flat = predictions.cpu().numpy().flatten()
        target_flat = targets.cpu().numpy().flatten()
        
        # Create valid mask (exclude ignore_index)
        valid_mask = target_flat != self.ignore_index
        pred_flat = pred_flat[valid_mask]
        target_flat = target_flat[valid_mask]
        
        # Ensure predictions are within valid range
        pred_flat = np.clip(pred_flat, 0, self.num_classes - 1)
        target_flat = np.clip(target_flat, 0, self.num_classes - 1)
        
        # Update confusion matrix
        cm = confusion_matrix(
            target_flat,
            pred_flat,
            labels=np.arange(self.num_classes)
        )
        self.confusion_matrix += cm
    
    def compute_iou(self):
        """
        Compute Intersection over Union (IoU) for each class
        
        Returns:
            per_class_iou: IoU for each class
            mean_iou: Mean IoU across all classes
            weighted_iou: Weighted IoU by class frequency
        """
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(axis=1) + 
                 self.confusion_matrix.sum(axis=0) - 
                 intersection)
        
        # Avoid division by zero
        valid = union > 0
        per_class_iou = np.zeros(self.num_classes)
        per_class_iou[valid] = intersection[valid] / union[valid]
        
        # Calculate mean IoU (only for classes present in dataset)
        mean_iou = per_class_iou[valid].mean() if valid.any() else 0.0
        
        # Calculate weighted IoU
        class_frequencies = self.confusion_matrix.sum(axis=1)
        total_pixels = class_frequencies.sum()
        if total_pixels > 0:
            weights = class_frequencies / total_pixels
            weighted_iou = (per_class_iou * weights).sum()
        else:
            weighted_iou = 0.0
        
        return per_class_iou, mean_iou, weighted_iou
    
    def compute_dice(self):
        """
        Compute Dice coefficient for each class
        
        Returns:
            per_class_dice: Dice score for each class
            mean_dice: Mean Dice across all classes
        """
        # Dice = 2*TP / (2*TP + FP + FN)
        intersection = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(axis=1) + 
                 self.confusion_matrix.sum(axis=0))
        
        # Avoid division by zero
        valid = union > 0
        per_class_dice = np.zeros(self.num_classes)
        per_class_dice[valid] = (2.0 * intersection[valid]) / union[valid]
        
        mean_dice = per_class_dice[valid].mean() if valid.any() else 0.0
        
        return per_class_dice, mean_dice
    
    def compute_precision_recall_f1(self):
        """
        Compute Precision, Recall, and F1 for each class
        
        Returns:
            per_class_precision: Precision for each class
            per_class_recall: Recall for each class
            per_class_f1: F1 score for each class
        """
        # Precision = TP / (TP + FP)
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp
        
        # Precision
        precision_denom = tp + fp
        per_class_precision = np.zeros(self.num_classes)
        valid_prec = precision_denom > 0
        per_class_precision[valid_prec] = tp[valid_prec] / precision_denom[valid_prec]
        
        # Recall
        recall_denom = tp + fn
        per_class_recall = np.zeros(self.num_classes)
        valid_rec = recall_denom > 0
        per_class_recall[valid_rec] = tp[valid_rec] / recall_denom[valid_rec]
        
        # F1 = 2 * (Precision * Recall) / (Precision + Recall)
        f1_denom = per_class_precision + per_class_recall
        per_class_f1 = np.zeros(self.num_classes)
        valid_f1 = f1_denom > 0
        per_class_f1[valid_f1] = (2.0 * per_class_precision[valid_f1] * 
                                   per_class_recall[valid_f1] / f1_denom[valid_f1])
        
        return per_class_precision, per_class_recall, per_class_f1
    
    def compute_all_metrics(self):
        """
        Compute all metrics
        
        Returns:
            Dictionary containing all metrics
        """
        per_class_iou, mean_iou, weighted_iou = self.compute_iou()
        per_class_dice, mean_dice = self.compute_dice()
        per_class_precision, per_class_recall, per_class_f1 = self.compute_precision_recall_f1()
        
        # Calculate macro and weighted averages for F1
        class_frequencies = self.confusion_matrix.sum(axis=1)
        total_pixels = class_frequencies.sum()
        
        # Macro F1 (equal weight to all classes)
        valid_classes = class_frequencies > 0
        macro_f1 = per_class_f1[valid_classes].mean() if valid_classes.any() else 0.0
        
        # Weighted F1 (weighted by class frequency)
        if total_pixels > 0:
            weights = class_frequencies / total_pixels
            weighted_f1 = (per_class_f1 * weights).sum()
        else:
            weighted_f1 = 0.0
        
        return {
            'per_class_iou': per_class_iou,
            'mean_iou': mean_iou,
            'weighted_iou': weighted_iou,
            'per_class_dice': per_class_dice,
            'mean_dice': mean_dice,
            'per_class_precision': per_class_precision,
            'per_class_recall': per_class_recall,
            'per_class_f1': per_class_f1,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'confusion_matrix': self.confusion_matrix
        }
    
    def get_rare_class_metrics(self, threshold=0.01):
        """
        Get metrics specifically for rare classes (frequency < threshold)
        
        Args:
            threshold: Frequency threshold to consider a class as rare
        
        Returns:
            Dictionary with rare class metrics
        """
        class_frequencies = self.confusion_matrix.sum(axis=1)
        total_pixels = class_frequencies.sum()
        
        if total_pixels == 0:
            return {'rare_class_count': 0, 'rare_class_f1': 0.0}
        
        class_freq_ratio = class_frequencies / total_pixels
        rare_classes = class_freq_ratio < threshold
        
        if not rare_classes.any():
            return {'rare_class_count': 0, 'rare_class_f1': 0.0}
        
        _, _, per_class_f1 = self.compute_precision_recall_f1()
        rare_class_f1 = per_class_f1[rare_classes].mean()
        
        rare_class_names = [self.class_names[i] for i in np.where(rare_classes)[0]]
        
        return {
            'rare_class_count': rare_classes.sum(),
            'rare_class_names': rare_class_names,
            'rare_class_f1': rare_class_f1,
            'rare_class_frequencies': class_freq_ratio[rare_classes]
        }
    
    def print_metrics(self, detailed=True):
        """
        Print all metrics in a formatted table
        
        Args:
            detailed: If True, print per-class metrics
        """
        metrics = self.compute_all_metrics()
        
        print("\n" + "="*80)
        print("SEGMENTATION METRICS SUMMARY")
        print("="*80)
        
        print(f"\nOverall Metrics:")
        print(f"  Mean IoU:      {metrics['mean_iou']:.4f}")
        print(f"  Weighted IoU:  {metrics['weighted_iou']:.4f}")
        print(f"  Mean Dice:     {metrics['mean_dice']:.4f}")
        print(f"  Macro F1:      {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1:   {metrics['weighted_f1']:.4f}")
        
        # Rare class metrics
        rare_metrics = self.get_rare_class_metrics()
        print(f"\nRare Class Performance (< 1% frequency):")
        print(f"  Rare Classes:  {rare_metrics['rare_class_count']}")
        print(f"  Rare F1 Score: {rare_metrics['rare_class_f1']:.4f}")
        
        if detailed:
            print("\n" + "-"*80)
            print("Per-Class Metrics:")
            print("-"*80)
            print(f"{'Class':<20} {'IoU':<8} {'Dice':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}")
            print("-"*80)
            
            for i in range(self.num_classes):
                class_name = self.class_names[i]
                if len(class_name) > 18:
                    class_name = class_name[:15] + "..."
                
                print(f"{class_name:<20} "
                      f"{metrics['per_class_iou'][i]:<8.4f} "
                      f"{metrics['per_class_dice'][i]:<8.4f} "
                      f"{metrics['per_class_precision'][i]:<8.4f} "
                      f"{metrics['per_class_recall'][i]:<8.4f} "
                      f"{metrics['per_class_f1'][i]:<8.4f}")
            
            print("="*80)


def calculate_miou(predictions, targets, num_classes, ignore_index=255):
    """
    Quick function to calculate mean IoU
    
    Args:
        predictions: (B, H, W) or (B, C, H, W) - predicted labels or logits
        targets: (B, H, W) - ground truth labels
        num_classes: Number of classes
        ignore_index: Index to ignore
    
    Returns:
        Mean IoU value
    """
    if len(predictions.shape) == 4:
        # Convert from logits to predictions
        predictions = torch.argmax(predictions, dim=1)
    
    metrics = SegmentationMetrics(num_classes, ignore_index)
    metrics.update(predictions, targets)
    _, mean_iou, _ = metrics.compute_iou()
    
    return mean_iou
