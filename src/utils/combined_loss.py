#!/usr/bin/env python3
"""
Combined Loss: Focal Loss + Dice Loss
Addresses class imbalance from two angles:
- Focal Loss: Down-weights easy examples, focuses on hard negatives
- Dice Loss: Directly optimizes IoU metric, better for minority classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.focal_loss import FocalLoss
from src.utils.dice_loss import DiceLoss


class CombinedLoss(nn.Module):
    """
    Combined Focal + Dice Loss for optimal class imbalance handling
    
    Loss = focal_weight * FocalLoss + dice_weight * DiceLoss
    
    Recommended weights:
    - focal_weight=0.7, dice_weight=0.3: Balanced (default)
    - focal_weight=0.8, dice_weight=0.2: More emphasis on hard examples
    - focal_weight=0.6, dice_weight=0.4: More emphasis on minority classes
    """
    
    def __init__(self, 
                 num_classes=12,
                 class_weights=None,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 focal_weight=0.7,
                 dice_weight=0.3,
                 smooth=1e-6):
        """
        Args:
            num_classes: Number of segmentation classes
            class_weights: Tensor of weights for each class
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter (focusing parameter)
            focal_weight: Weight for focal loss component
            dice_weight: Weight for dice loss component
            smooth: Smoothing factor for numerical stability
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        # Verify weights sum to 1
        assert abs(focal_weight + dice_weight - 1.0) < 1e-6, \
            f"focal_weight ({focal_weight}) + dice_weight ({dice_weight}) must equal 1.0"
        
        # Initialize component losses
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction='mean',
            class_weights=class_weights
        )
        
        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            smooth=smooth
        )
    
    def forward(self, pred, target):
        """
        Calculate combined loss
        
        Args:
            pred: Model output logits (B, C, H, W)
            target: Ground truth masks (B, H, W)
        
        Returns:
            Combined loss (scalar)
        """
        # Calculate component losses
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        # Combine losses
        combined = self.focal_weight * focal + self.dice_weight * dice
        
        return combined


class AdaptiveCombinedLoss(nn.Module):
    """
    Adaptive Combined Loss: Adjusts weights during training
    
    Early epochs: Emphasize focal loss (focus on hard examples)
    Later epochs: Emphasize dice loss (fine-tune predictions)
    
    This helps the model first learn good features, then optimize IoU metric
    """
    
    def __init__(self,
                 num_classes=12,
                 class_weights=None,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 total_epochs=50,
                 smooth=1e-6):
        """
        Args:
            num_classes: Number of segmentation classes
            class_weights: Tensor of weights for each class
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
            total_epochs: Total training epochs (for scheduling)
            smooth: Smoothing factor
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
        # Component losses
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction='mean',
            class_weights=class_weights
        )
        
        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            smooth=smooth
        )
    
    def set_epoch(self, epoch):
        """Update current epoch for adaptive scheduling"""
        self.current_epoch = epoch
    
    def _get_weights(self):
        """
        Calculate adaptive weights based on epoch
        
        Focal weight: 0.8 → 0.6 (decreases over time)
        Dice weight: 0.2 → 0.4 (increases over time)
        """
        progress = self.current_epoch / self.total_epochs
        
        # Smooth transition from focal-heavy to dice-heavy
        focal_weight = 0.8 - (0.2 * progress)
        dice_weight = 0.2 + (0.2 * progress)
        
        return focal_weight, dice_weight
    
    def forward(self, pred, target):
        """Calculate adaptive combined loss"""
        focal_weight, dice_weight = self._get_weights()
        
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        combined = focal_weight * focal + dice_weight * dice
        
        return combined
