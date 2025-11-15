#!/usr/bin/env python3
"""
Focal Loss Implementation for Handling Class Imbalance
Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
https://arxiv.org/abs/1708.02002
"""

# import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in segmentation tasks
    
    The focal loss applies a modulating term to the cross entropy loss in order to
    focus learning on hard negative examples.
    
    Args:
        alpha (float or list): Weighting factor in range (0,1) to balance
            positive vs negative examples or a list of weights for each class.
            Default: 0.25
        gamma (float): Exponent of the modulating factor (1 - p_t) ^ gamma.
            Default: 2.0
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'
        class_weights (Tensor, optional): Manual rescaling weight given to each class.
            If given, has to be a Tensor of size C
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', class_weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions (logits) for each example.
                    Shape: (batch_size, num_classes, height, width)
            targets: A long tensor of the same shape as inputs.
                    Stores the class index for each pixel.
                    Shape: (batch_size, height, width)
        
        Returns:
            focal_loss: A scalar loss value
        """
        # Reshape for easier processing
        b, c, h, w = inputs.shape
        inputs_flat = inputs.permute(0, 2, 3, 1).reshape(-1, c)  # (N*H*W, C)
        targets_flat = targets.reshape(-1)  # (N*H*W,)
        
        # Get probabilities
        p = F.softmax(inputs_flat, dim=1)
        
        # Get class probabilities
        class_mask = F.one_hot(targets_flat, c).float()  # (N*H*W, C)
        probs = (p * class_mask).sum(dim=1)  # (N*H*W,)
        
        # Calculate focal loss
        log_p = F.log_softmax(inputs_flat, dim=1)
        log_p = (log_p * class_mask).sum(dim=1)  # (N*H*W,)
        
        # Focal loss = -alpha * (1-p)^gamma * log(p)
        focal_weight = (1 - probs) ** self.gamma
        focal_loss = -self.alpha * focal_weight * log_p
        
        # Apply class weights if provided
        if self.class_weights is not None:
            class_weights = self.class_weights.to(targets.device)
            weight_t = class_weights[targets_flat]
            focal_loss = focal_loss * weight_t
        
        # Apply reduction
        if self.reduction == 'none':
            return focal_loss.reshape(b, h, w)
        elif self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss combining Focal Loss with class weights
    Best for extreme class imbalance scenarios
    
    Args:
        alpha (float): Focal loss alpha parameter (0.25 recommended)
        gamma (float): Focal loss gamma parameter (2.0 recommended)
        class_weights (Tensor): Class weights for balancing
        reduction (str): Reduction type ('mean', 'sum', 'none')
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.focal_loss = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            reduction=reduction,
            class_weights=class_weights
        )
    
    def forward(self, inputs, targets):
        return self.focal_loss(inputs, targets)


def create_focal_loss(num_classes, class_weights=None, alpha=0.25, gamma=2.0, device='cpu'):
    """
    Factory function to create Focal Loss with optional class weights
    
    Args:
        num_classes (int): Number of classes
        class_weights (Tensor, optional): Class weights for imbalance handling
        alpha (float): Focal loss alpha parameter
        gamma (float): Focal loss gamma parameter
        device (str): Device to place weights on
    
    Returns:
        FocalLoss: Configured focal loss function
    """
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    return FocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction='mean',
        class_weights=class_weights
    )
