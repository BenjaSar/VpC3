"""Evaluation metrics for segmentation."""
import torch
import numpy as np
from typing import Dict


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """Calculate IoU for each class."""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())
    
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0
    
    return {'iou_per_class': ious, 'mean_iou': mean_iou}


def calculate_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate pixel accuracy."""
    correct = (pred == target).sum()
    total = target.numel()
    return (correct / total).item()
