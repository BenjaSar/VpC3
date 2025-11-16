#!/usr/bin/env python3
"""
Simplified Hybrid ViT-CNN Segmentation Model
Combines ViT encoder (global context) with CNN decoder (spatial details)
Removes complex skip connections for stability
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings"""
    def __init__(self, img_size=512, patch_size=16, in_channels=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.n_patches_side = img_size // patch_size
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class TransformerEncoderLayer(nn.Module):
    """Standard Transformer encoder layer with self-attention"""
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


class DecoderBlock(nn.Module):
    """Simple decoder block: Upsample -> Conv -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels, upsample_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=upsample_factor, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class HybridViTCNNSegmentation(nn.Module):
    """
    Simplified Hybrid ViT-CNN Segmentation Model
    - ViT Encoder: Global context with attention
    - CNN Decoder: Simple progressive upsampling (no skip connections for stability)
    """
    def __init__(self, img_size=512, patch_size=16, in_channels=3, n_classes=12,
                 embed_dim=384, n_encoder_layers=12, n_heads=6, mlp_ratio=4.0, 
                 dropout=0.25, skip_connections=False):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_classes = n_classes
        
        # ==================== ViT ENCODER ====================
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_encoder_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # ==================== CNN DECODER ====================
        # Simplified decoder without skip connections
        self.decoder = nn.Sequential(
            # 32x32 -> 64x64
            DecoderBlock(embed_dim, 256, upsample_factor=2),
            # 64x64 -> 128x128
            DecoderBlock(256, 128, upsample_factor=2),
            # 128x128 -> 256x256
            DecoderBlock(128, 64, upsample_factor=2),
            # 256x256 -> 512x512
            DecoderBlock(64, 32, upsample_factor=2),
        )
        
        # Final segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, n_classes, 1)
        )
        
    def forward(self, x):
        B = x.shape[0]
        
        # ==================== ViT ENCODER ====================
        x_embed = self.patch_embed(x)
        x_embed = x_embed + self.pos_embed
        
        for layer in self.encoder_layers:
            x_embed = layer(x_embed)
        
        x_embed = self.norm(x_embed)
        
        # Reshape to spatial format (B, embed_dim, 32, 32)
        n_patches_side = self.patch_embed.n_patches_side
        x_spatial = x_embed.transpose(1, 2)
        x_spatial = x_spatial.reshape(B, -1, n_patches_side, n_patches_side)
        
        # ==================== CNN DECODER (simplified, no skip connections) ====================
        x_out = self.decoder(x_spatial)
        
        # ==================== SEGMENTATION HEAD ====================
        output = self.seg_head(x_out)
        
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
