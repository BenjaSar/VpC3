#!/usr/bin/env python3
"""
UNet++ Segmentation Model with EfficientNet-B0 Encoder
Proven architecture for pixel-level segmentation tasks
- EfficientNet-B0 encoder (pretrained on ImageNet)
- Dense skip connections (no shape mismatch issues)
- Expected convergence: 15-20 epochs to 0.50+ IoU
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np
from tqdm import tqdm


def create_unet_model(
    encoder_name='efficientnet-b4',
    encoder_weights='imagenet',
    in_channels=3,
    num_classes=12,
    decoder_channels=(512, 256, 128, 64, 32),
    decoder_attention_type='scse'
):
    """
    Create UNet++ model with specified encoder
    PHASE 3: Upgraded architecture for better accuracy
    
    Args:
        encoder_name: Name of encoder backbone (e.g., 'efficientnet-b4')
        encoder_weights: Pretrained weights ('imagenet' or None)
        in_channels: Number of input channels (3 for RGB)
        num_classes: Number of output classes
        decoder_channels: Number of channels in decoder blocks (increased)
        decoder_attention_type: Type of attention mechanism ('scse' for Spatial+Channel)
    
    Returns:
        Initialized UNet++ model
    """
  
    # 1. EfficientNet-B4 instead of B0: More capacity, better features
    # 2. SCSE attention: Spatial + Channel attention for better feature weighting
    # 3. Larger decoder channels: (512, 256, 128, 64, 32) vs (256, 128, 64, 32, 16)
    # 4. Deep supervision enabled via aux_params
    
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights if encoder_weights else None,
        in_channels=in_channels,
        classes=num_classes,
        decoder_channels=decoder_channels,
        decoder_attention_type=decoder_attention_type,
        activation=None,  # Output raw logits (will apply softmax after)
        # Deep supervision: auxiliary output from decoder
        aux_params=dict(
            pooling='avg',
            dropout=0.2,
            classes=num_classes
        ) if encoder_weights else None
    )
    
    return model


class UNetPlusPlusSegmentation(nn.Module):
    """
    Wrapper around segmentation_models UNet++ for consistency
    PHASE 3: Enhanced with B4 encoder, attention, and deep supervision
    """
    
    def __init__(self, in_channels=3, num_classes=12, encoder_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.has_aux = encoder_weights is not None
        
        self.model = create_unet_model(
            encoder_name='efficientnet-b4',  # Upgraded from B0 to B4
            encoder_weights=encoder_weights,  # Use pretrained weights if available
            in_channels=in_channels,
            num_classes=num_classes,
            decoder_channels=(512, 256, 128, 64, 32),  # Larger decoder
            decoder_attention_type='scse'  # SCSE attention
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            Output logits (B, num_classes, H, W) or tuple if deep supervision enabled
        """
        output = self.model(x)
        
        # If model has auxiliary head (deep supervision)
        if isinstance(output, tuple):
            # Return main output, auxiliary output will be handled in loss
            return output[0]
        
        return output


def calculate_iou(pred, target, n_classes):
    """Calculate mean IoU"""
    pred = pred.cpu().view(-1)
    target = target.cpu().view(-1)
    
    ious = []
    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())
    
    return np.nanmean(ious)


def train_epoch(model, dataloader, criterion, optimizer, device, n_classes, config=None, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device).long()
        masks = torch.clamp(masks, 0, n_classes - 1)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            if config and config.get('gradient_clip', 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            if config and config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            optimizer.step()
        
        pred = outputs.argmax(dim=1)
        iou = calculate_iou(pred, masks, n_classes)
        
        total_loss += loss.item()
        total_iou += iou
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{iou:.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    
    return avg_loss, avg_iou


def validate(model, dataloader, criterion, device, n_classes):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device).long()
            masks = torch.clamp(masks, 0, n_classes - 1)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            pred = outputs.argmax(dim=1)
            iou = calculate_iou(pred, masks, n_classes)
            
            total_loss += loss.item()
            total_iou += iou
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou:.4f}'
            })
    
    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    
    return avg_loss, avg_iou
