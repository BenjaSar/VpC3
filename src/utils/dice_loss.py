#!/usr/bin/env python3
"""
Dice Loss Implementation for Semantic Segmentation
Especially effective for handling class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation
    Better at handling class imbalance than CrossEntropy alone
    
    Dice coefficient = 2 * (intersection) / (union + intersection)
    Dice Loss = 1 - Dice coefficient
    """
    
    def __init__(self, num_classes=12, smooth=1e-6, ignore_index=None):
        """
        Args:
            num_classes: Number of segmentation classes
            smooth: Smoothing factor to prevent division by zero
            ignore_index: Class index to ignore (e.g., -1 or 255)
        """
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        """
        Calculate Dice Loss
        
        Args:
            pred: Model output logits (B, C, H, W)
            target: Ground truth masks (B, H, W) with class indices
        
        Returns:
            Dice loss (scalar)
        """
        # Convert logits to probabilities
        pred_probs = F.softmax(pred, dim=1)
        
        # Convert target to one-hot encoding
        target_one_hot = F.one_hot(target.long(), num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Calculate intersection
        intersection = (pred_probs * target_one_hot).sum(dim=(2, 3))
        
        # Calculate union
        pred_sum = pred_probs.sum(dim=(2, 3))
        target_sum = target_one_hot.sum(dim=(2, 3))
        union = pred_sum + target_sum
        
        # Calculate Dice coefficient per class
        dice_coeff = (2 * intersection + self.smooth) / (union + self.smooth)
        
        # Average Dice coefficient across classes
        dice_loss = 1 - dice_coeff.mean()
        
        return dice_loss


class WeightedDiceLoss(nn.Module):
    """
    Weighted Dice Loss with class weights
    Allows prioritizing minority classes
    """
    
    def __init__(self, num_classes=12, class_weights=None, smooth=1e-6):
        """
        Args:
            num_classes: Number of segmentation classes
            class_weights: Tensor of shape (num_classes,) with class weights
            smooth: Smoothing factor
        """
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        
        if class_weights is None:
            class_weights = torch.ones(num_classes)
        
        self.register_buffer('class_weights', class_weights)
    
    def forward(self, pred, target):
        """Calculate weighted Dice loss"""
        pred_probs = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target.long(), num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        intersection = (pred_probs * target_one_hot).sum(dim=(2, 3))
        pred_sum = pred_probs.sum(dim=(2, 3))
        target_sum = target_one_hot.sum(dim=(2, 3))
        union = pred_sum + target_sum
        
        dice_coeff = (2 * intersection + self.smooth) / (union + self.smooth)
        
        # Apply class weights
        weighted_dice = (1 - dice_coeff) * self.class_weights.unsqueeze(0)
        
        return weighted_dice.mean()
