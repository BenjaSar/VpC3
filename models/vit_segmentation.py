#!/usr/bin/env python3
"""
Training Script for ViT-Small Floor Plan Segmentation
Integrates with the project's dataset and preprocessing pipeline
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
# import torch.optim as optim
from torch.cuda.amp import autocast #, GradScaler
import numpy as np
from tqdm import tqdm
# import json
# from datetime import datetime

# Import project modules
# from src.data.dataset import create_dataloaders
from src.utils.logging_config import setup_logging

logger = setup_logging()


# ==================== Model Architecture ====================

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=512, patch_size=32, in_channels=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, 3, 512, 512)
        x = self.proj(x)  # (B, 384, 16, 16)
        x = x.flatten(2)  # (B, 384, 256)
        x = x.transpose(1, 2)  # (B, 256, 384)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim=384, n_heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim=384, n_heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class SegmentationHead(nn.Module):
    def __init__(self, embed_dim=384, patch_size=32, img_size=512, n_classes=34):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.n_patches_side = img_size // patch_size
        
        self.conv_transpose = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, n_classes, kernel_size=4, stride=2, padding=1)
        )
        
    def forward(self, x):
        B = x.shape[0]
        x = x.transpose(1, 2)
        x = x.reshape(B, -1, self.n_patches_side, self.n_patches_side)
        x = self.conv_transpose(x)
        return x


class ViTSegmentation(nn.Module):
    def __init__(self, img_size=512, patch_size=32, in_channels=3, n_classes=34,
                 embed_dim=384, n_encoder_layers=12, n_decoder_layers=3, 
                 n_heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_decoder_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.seg_head = SegmentationHead(embed_dim, patch_size, img_size, n_classes)
        
    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        for layer in self.encoder_layers:
            x = layer(x)
        
        for layer in self.decoder_layers:
            x = layer(x)
        
        x = self.norm(x)
        x = self.seg_head(x)
        
        return x


# ==================== Training Functions ====================

def calculate_iou(pred, target, n_classes):
    """Calculate mean IoU"""
    # Move to CPU to avoid CUDA assertions
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


def train_epoch(model, dataloader, criterion, optimizer, device, n_classes, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device).long()  # Convert to Long type
        
        # Clip mask values to valid range [0, n_classes-1]
        masks = torch.clamp(masks, 0, n_classes - 1)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        # Calculate metrics
        pred = outputs.argmax(dim=1)
        iou = calculate_iou(pred, masks, outputs.shape[1])
        
        total_loss += loss.item()
        total_iou += iou
        
        # Update progress bar
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
            masks = batch['mask'].to(device).long()  # Convert to Long type
            
            # Clip mask values to valid range [0, n_classes-1]
            masks = torch.clamp(masks, 0, n_classes - 1)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            pred = outputs.argmax(dim=1)
            iou = calculate_iou(pred, masks, outputs.shape[1])
            
            total_loss += loss.item()
            total_iou += iou
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou:.4f}'
            })
    
    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    
    return avg_loss, avg_iou


# Note: Training is handled by scripts/train.py, not this module
# This module provides the model architecture and utility functions
