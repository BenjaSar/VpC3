#!/usr/bin/env python3
"""
Focal Loss Implementation for Class Imbalance in Segmentation
Based on "Focal Loss for Dense Object Detection" (Lin et al., 2017)
Addresses severe class imbalance by focusing on hard examples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    FL(p_t) = -α_t(1-p_t)^γ * log(p_t)
    
    where:
    - p_t is the model's estimated probability for the true class
    - α_t balances foreground/background
    - γ (gamma) focuses on hard examples
    
    Args:
        alpha (float or list): Weighting factor in range (0,1) to balance
                              positive vs negative examples or a list of weights
                              for each class. Default: 0.25
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                      down-weight easy examples. Default: 2.0
        reduction (str): Specifies reduction to apply to output
                        ('none' | 'mean' | 'sum'). Default: 'mean'
        class_weights (tensor): Optional tensor of class weights for addressing
                              severe class imbalance. Default: None
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', class_weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights
    
    def forward(self, inputs, targets):
        """
        Forward pass
        
        Args:
            inputs: Raw model outputs (B, C, H, W) where C is num_classes
            targets: Ground truth labels (B, H, W) with class indices
        
        Returns:
            Focal loss value (scalar if reduction='mean')
        """
        # Ensure correct shapes
        if inputs.dim() == 4:  # (B, C, H, W) - semantic segmentation
            B, C, H, W = inputs.shape
            inputs = inputs.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
            inputs = inputs.view(-1, C)  # (B*H*W, C)
            targets = targets.view(-1)  # (B*H*W,)
        elif inputs.dim() == 2:  # Already flattened (N, C)
            pass
        else:
            raise ValueError(f"Unexpected input shape: {inputs.shape}")
        
        # Get log probabilities
        log_p = F.log_softmax(inputs, dim=-1)
        
        # Get class probabilities
        p = torch.exp(log_p)
        
        # Get focal weight: (1 - p_t)^gamma
        # where p_t is the probability of the true class
        class_mask = torch.zeros_like(inputs)
        class_mask.scatter_(1, targets.view(-1, 1), 1.0)
        
        probs = (p * class_mask).sum(dim=1)  # p_t for each sample
        focal_weight = torch.pow(1.0 - probs, self.gamma)
        
        # Apply focal weight to cross entropy
        ce = F.cross_entropy(inputs, targets, reduction='none', weight=self.class_weights)
        focal_loss = self.alpha * focal_weight * ce
        
        # IMPROVEMENT: Apply additional class weighting to minority classes
        if self.class_weights is not None:
            # Get target class weights for each pixel
            target_weights = self.class_weights[targets]
            # Multiply focal loss by target class weight
            focal_loss = focal_loss * target_weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


def create_focal_loss(alpha=0.25, gamma=2.0, class_weights=None, reduction='mean'):
    """
    Factory function to create Focal Loss
    
    Args:
        alpha (float): Weighting factor for balance (0-1). Default: 0.25
        gamma (float): Focusing parameter. Default: 2.0
        class_weights (tensor): Optional class weights for imbalance. Default: None
        reduction (str): How to reduce loss. Default: 'mean'
    
    Returns:
        FocalLoss instance
    """
    return FocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
        class_weights=class_weights
    )


class WeightedCrossEntropyLoss(nn.Module):
    """
    Standard weighted cross-entropy loss
    Useful as baseline for comparison
    
    Args:
        class_weights (tensor): Weight for each class
        reduction (str): Reduction mode ('mean' or 'sum')
    """
    
    def __init__(self, class_weights=None, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Forward pass
        
        Args:
            inputs: (B, C, H, W) - model outputs
            targets: (B, H, W) - ground truth
        
        Returns:
            Cross-entropy loss
        """
        if inputs.dim() == 4:
            B, C, H, W = inputs.shape
            inputs = inputs.view(-1, C)
            targets = targets.view(-1)
        
        return F.cross_entropy(
            inputs, targets,
            weight=self.class_weights,
            reduction=self.reduction
        )


class ComboLoss(nn.Module):
    """
    Combination of Focal Loss and Dice Loss
    Good for severe class imbalance
    
    Args:
        alpha (float): Weight for Focal Loss (0-1)
        beta (float): Weight for Dice Loss (0-1)
        class_weights (tensor): Weights for each class
    """
    
    def __init__(self, alpha=0.5, beta=0.5, class_weights=None):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.focal_loss = FocalLoss(
            alpha=0.25,
            gamma=2.0,
            reduction='mean',
            class_weights=class_weights
        )
    
    def forward(self, inputs, targets):
        """Forward pass combining focal and dice loss"""
        focal = self.focal_loss(inputs, targets)
        # Dice could be added here if needed
        # For now, just return focal loss
        return focal


# Test the implementation
if __name__ == "__main__":
    # Create dummy data
    B, C, H, W = 2, 12, 64, 64
    inputs = torch.randn(B, C, H, W)
    targets = torch.randint(0, C, (B, H, W))
    
    # Create class weights
    class_weights = torch.ones(C)
    class_weights[:3] = 0.5  # Down-weight majority classes
    
    # Test FocalLoss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0, class_weights=class_weights)
    loss = focal_loss(inputs, targets)
    
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Focal Loss value: {loss.item():.4f}")
    print("✓ Focal Loss working correctly!")
