"""
Advanced Loss Functions for Segmentation
Implements Focal Loss, Dice Loss, and Combined Loss for handling class imbalance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses training on hard examples and down-weights easy examples.
    
    Args:
        alpha: Weighting factor in range (0,1) to balance positive/negative examples
        gamma: Exponent of the modulating factor (1 - p_t)^gamma
        ignore_index: Specifies a target value that is ignored
        reduction: Specifies the reduction to apply to the output
    """
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C, H, W) - raw logits from model
            targets: (B, H, W) - ground truth labels
        """
        # Get valid mask (excluding ignore_index)
        valid_mask = (targets != self.ignore_index)
        
        # Flatten for processing
        inputs = inputs.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        inputs = inputs.view(-1, inputs.size(-1))  # (B*H*W, C)
        targets_flat = targets.view(-1)  # (B*H*W)
        valid_mask_flat = valid_mask.view(-1)  # (B*H*W)
        
        # Apply valid mask
        inputs = inputs[valid_mask_flat]
        targets_flat = targets_flat[valid_mask_flat]
        
        # Clamp targets to valid range [0, num_classes-1]
        num_classes = inputs.size(1)
        targets_flat = targets_flat.clamp(0, num_classes - 1)
        
        # Calculate cross entropy
        ce_loss = F.cross_entropy(inputs, targets_flat, reduction='none')
        
        # Get probabilities
        p = F.softmax(inputs, dim=1)
        # Use gather to safely index the target class probabilities
        p_t = p.gather(1, targets_flat.long().unsqueeze(1)).squeeze(1)
        
        # Calculate focal loss
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    Particularly effective for handling class imbalance and improving boundary segmentation.
    
    Args:
        smooth: Smoothing constant to avoid division by zero
        ignore_index: Specifies a target value that is ignored
        reduction: Specifies the reduction to apply to the output
    """
    def __init__(self, smooth=1.0, ignore_index=255, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C, H, W) - raw logits from model
            targets: (B, H, W) - ground truth labels
        """
        # Get valid mask
        valid_mask = (targets != self.ignore_index)
        
        # Get softmax probabilities
        probs = F.softmax(inputs, dim=1)
        num_classes = inputs.size(1)
        
        # Simplified Dice loss - compute on flattened data to reduce memory
        targets_clamped = targets.clamp(0, num_classes - 1).long()
        
        # Flatten everything
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, num_classes)
        targets_flat = targets_clamped.reshape(-1)
        valid_mask_flat = valid_mask.reshape(-1)
        
        # Create one-hot for targets (on CPU to save GPU memory, then move back if needed)
        targets_one_hot = F.one_hot(targets_flat.cpu(), num_classes).float()
        if probs_flat.is_cuda:
            targets_one_hot = targets_one_hot.to(probs_flat.device)
        
        # Apply valid mask
        valid_mask_one_hot = valid_mask_flat.view(-1, 1).float()
        probs_flat = probs_flat * valid_mask_one_hot
        targets_one_hot = targets_one_hot * valid_mask_one_hot
        
        # Compute Dice per class
        intersection = (probs_flat * targets_one_hot).sum(dim=0)
        union = probs_flat.sum(dim=0) + targets_one_hot.sum(dim=0)
        
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = (1.0 - dice_score).mean()
        
        return dice_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function that combines Focal Loss, Dice Loss, and CrossEntropy Loss
    with configurable weights.
    
    Args:
        focal_weight: Weight for focal loss component
        dice_weight: Weight for dice loss component
        ce_weight: Weight for cross entropy loss component
        class_weights: Tensor of class weights for handling imbalance
        focal_alpha: Alpha parameter for Focal Loss
        focal_gamma: Gamma parameter for Focal Loss
        ignore_index: Specifies a target value that is ignored
    """
    def __init__(self, focal_weight=1.0, dice_weight=1.0, ce_weight=1.0,
                 class_weights=None, focal_alpha=0.25, focal_gamma=2.0,
                 ignore_index=255):
        super(CombinedLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            ignore_index=ignore_index
        )
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(
                weight=class_weights,
                ignore_index=ignore_index
            )
        else:
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C, H, W) - raw logits from model
            targets: (B, H, W) - ground truth labels
        """
        loss = 0.0
        
        if self.focal_weight > 0:
            focal = self.focal_loss(inputs, targets)
            loss += self.focal_weight * focal
        
        if self.dice_weight > 0:
            dice = self.dice_loss(inputs, targets)
            loss += self.dice_weight * dice
        
        if self.ce_weight > 0:
            ce = self.ce_loss(inputs, targets)
            loss += self.ce_weight * ce
        
        return loss


def calculate_log_scaled_weights(class_counts, num_classes, beta=0.999):
    """
    Calculate log-scaled class weights to handle severe class imbalance.
    Uses effective number of samples approach with logarithmic scaling.
    
    Args:
        class_counts: Array of pixel counts per class
        num_classes: Total number of classes
        beta: Hyperparameter for effective number calculation (0.9-0.999)
    
    Returns:
        Tensor of normalized class weights
    """
    # Calculate effective number of samples
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / np.array(effective_num)
    
    # Normalize weights
    weights = weights / np.sum(weights) * num_classes
    
    # Apply log scaling to prevent extreme weights
    weights = np.log(weights + 1.0)
    
    # Normalize again
    weights = weights / np.max(weights)
    
    return torch.tensor(weights, dtype=torch.float32)


def calculate_class_weights_from_dataloader(dataloader, num_classes, device='cpu'):
    """
    Calculate class weights from a dataloader by counting pixel occurrences.
    
    Args:
        dataloader: PyTorch DataLoader
        num_classes: Number of classes
        device: Device to put weights on
    
    Returns:
        Tensor of class weights
    """
    class_counts = np.zeros(num_classes, dtype=np.float64)
    
    print("Calculating class weights from dataset...")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % 50 == 0:
            print(f"Processing batch {batch_idx}/{len(dataloader)}")
        
        # Handle both dict and tuple formats
        if isinstance(batch, dict):
            masks = batch['mask']
        else:
            masks = batch[1]
        
        for mask in masks:
            mask_np = mask.cpu().numpy()
            # Count pixels for each class, excluding invalid labels
            for class_idx in range(num_classes):
                class_counts[class_idx] += np.sum(mask_np == class_idx)
    
    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1.0)
    
    # Calculate log-scaled weights
    weights = calculate_log_scaled_weights(class_counts, num_classes)
    
    return weights.to(device)
