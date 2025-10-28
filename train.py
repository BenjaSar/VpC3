#!/usr/bin/env python3
"""
Training Script for ViT-Small Floor Plan Segmentation
Integrates with the project's dataset and preprocessing pipeline
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Import project modules
from data.dataset import create_dataloaders
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


# ==================== Main Training Loop ====================

def main():
    # Configuration
    CONFIG = {
        # Data
        'images_dir': 'data/processed/images',
        'masks_dir': 'data/processed/annotations',
        'batch_size': 4,
        'num_workers': 0,  # Use 0 for Windows
        
        # Model
        'img_size': 512,
        'patch_size': 32,
        'n_classes': 256,
        'embed_dim': 384,
        'n_encoder_layers': 12,
        'n_decoder_layers': 3,
        'n_heads': 6,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        
        # Training
        'num_epochs': 100,
        #'num_epochs': 10,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'mixed_precision': True,
        
        # Checkpointing
        'checkpoint_dir': 'models/checkpoints',
        'save_frequency': 5
    }
    
    # Create checkpoint directory
    checkpoint_dir = Path(CONFIG['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = checkpoint_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=4)
    logger.info(f"Configuration saved to {config_path}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        images_dir=CONFIG['images_dir'],
        masks_dir=CONFIG['masks_dir'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        image_size=CONFIG['img_size'],
        num_classes=CONFIG['n_classes']
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Create model
    logger.info("Creating model...")
    model = ViTSegmentation(
        img_size=CONFIG['img_size'],
        patch_size=CONFIG['patch_size'],
        in_channels=3,
        n_classes=CONFIG['n_classes'],
        embed_dim=CONFIG['embed_dim'],
        n_encoder_layers=CONFIG['n_encoder_layers'],
        n_decoder_layers=CONFIG['n_decoder_layers'],
        n_heads=CONFIG['n_heads'],
        mlp_ratio=CONFIG['mlp_ratio'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params / 1e6:.2f}M")
    logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=CONFIG['num_epochs']
    )
    
    # Mixed precision
    scaler = GradScaler() if CONFIG['mixed_precision'] and torch.cuda.is_available() else None
    if scaler:
        logger.info("Using mixed precision training")
    
    # Training loop
    best_val_iou = 0.0
    history = {
        'train_loss': [],
        'train_iou': [],
        'val_loss': [],
        'val_iou': [],
        'lr': []
    }
    
    logger.info("="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)
    
    for epoch in range(CONFIG['num_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        logger.info("-" * 80)
        
        # Train
        train_loss, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, device, CONFIG['n_classes'], scaler
        )
        logger.info(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        
        # Validate
        val_loss, val_iou = validate(model, val_loader, criterion, device, CONFIG['n_classes'])
        logger.info(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Learning Rate: {current_lr:.6f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['lr'].append(current_lr)
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_model_path = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'config': CONFIG
            }, best_model_path)
            logger.info(f"Saved best model with IoU: {val_iou:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % CONFIG['save_frequency'] == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'history': history,
                'config': CONFIG
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model and history
    final_model_path = checkpoint_dir / 'final_model.pth'
    torch.save({
        'epoch': CONFIG['num_epochs'] - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
        'config': CONFIG
    }, final_model_path)
    
    history_path = checkpoint_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    logger.info("="*80)
    logger.info("TRAINING COMPLETED!")
    logger.info("="*80)
    logger.info(f"Best Val IoU: {best_val_iou:.4f}")
    logger.info(f"Models saved to: {checkpoint_dir}")
    logger.info(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\n\nTraining failed: {e}", exc_info=True)
