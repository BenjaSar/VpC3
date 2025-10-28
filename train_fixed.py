#!/usr/bin/env python3
"""
FIXED Training Script for ViT-Small Floor Plan Segmentation
Addresses severe class imbalance with weighted loss function
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from collections import Counter

# Import project modules
from data.dataset import create_dataloaders
from src.utils.logging_config import setup_logging
from train import ViTSegmentation, calculate_iou, train_epoch, validate

logger = setup_logging()


def calculate_class_weights(dataloader, num_classes=34, device='cpu'):
    """
    Calculate class weights based on inverse frequency
    This helps the model learn minority classes
    """
    logger.info("Calculating class weights from training data...")
    
    class_counts = Counter()
    total_pixels = 0
    
    for batch in tqdm(dataloader, desc="Analyzing class distribution"):
        masks = batch['mask'].numpy()
        for mask in masks:
            unique, counts = np.unique(mask, return_counts=True)
            for cls, count in zip(unique, counts):
                class_counts[int(cls)] += int(count)
                total_pixels += int(count)
    
    # Calculate weights
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)  # Avoid division by zero
        # Inverse frequency with smoothing
        weight = total_pixels / (num_classes * count)
        # Cap maximum weight to avoid extreme values
        weight = min(weight, 100.0)
        weights.append(weight)
    
    # Normalize weights
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * num_classes  # Normalize to average=1
    
    logger.info("\nClass Weight Statistics:")
    logger.info(f"  Min weight: {weights.min():.4f}")
    logger.info(f"  Max weight: {weights.max():.4f}")
    logger.info(f"  Mean weight: {weights.mean():.4f}")
    
    # Show weight distribution
    logger.info("\nClass weights (first 10 classes):")
    for i in range(min(10, num_classes)):
        count = class_counts.get(i, 0)
        pct = (count / total_pixels * 100) if total_pixels > 0 else 0
        logger.info(f"  Class {i}: weight={weights[i]:.3f}, pixels={count:,} ({pct:.2f}%)")
    
    return weights.to(device)


def train_epoch_with_class_iou(model, dataloader, criterion, optimizer, device, n_classes, scaler=None):
    """
    Train for one epoch with per-class IoU tracking
    """
    model.train()
    total_loss = 0.0
    class_ious = {i: [] for i in range(n_classes)}
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device).long()
        masks = torch.clamp(masks, 0, n_classes - 1)
        
        optimizer.zero_grad()
        
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
        
        # Calculate per-class IoU
        pred = outputs.argmax(dim=1)
        for cls in range(n_classes):
            pred_mask = (pred == cls)
            true_mask = (masks == cls)
            intersection = (pred_mask & true_mask).sum().float()
            union = (pred_mask | true_mask).sum().float()
            
            if union > 0:
                iou = (intersection / union).item()
                class_ious[cls].append(iou)
        
        total_loss += loss.item()
        
        # Update progress bar
        mean_iou = np.mean([np.mean(ious) if ious else 0 for ious in class_ious.values()])
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mIoU': f'{mean_iou:.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    
    # Calculate mean IoU per class
    per_class_iou = {cls: np.mean(ious) if ious else 0.0 
                     for cls, ious in class_ious.items()}
    mean_iou = np.mean(list(per_class_iou.values()))
    
    # Count classes with non-zero IoU
    active_classes = sum(1 for iou in per_class_iou.values() if iou > 0.01)
    
    return avg_loss, mean_iou, per_class_iou, active_classes


def main():
    # Configuration with improvements for class imbalance
    CONFIG = {
        # Data
        'images_dir': 'data/processed/images',
        'masks_dir': 'data/processed/annotations',
        'batch_size': 4,  # Reduced for stability
        'num_workers': 0,
        
        # Model - Slightly smaller for better convergence
        'img_size': 512,
        'patch_size': 32,
        'n_classes': 256,
        'embed_dim': 384,
        'n_encoder_layers': 12,
        'n_decoder_layers': 3,
        'n_heads': 6,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        
        # Training - Adjusted for class imbalance
        'num_epochs': 150,  # More epochs needed with class weights
        'learning_rate': 5e-5,  # Lower LR for stability
        'weight_decay': 0.01,
        'mixed_precision': True,
        
        # Loss function
        'use_class_weights': True,  # NEW: Enable class weighting
        'label_smoothing': 0.1,  # NEW: Label smoothing helps
        
        # Checkpointing
        'checkpoint_dir': 'models/checkpoints_fixed',
        'save_frequency': 10
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
    
    # Calculate class weights
    if CONFIG['use_class_weights']:
        class_weights = calculate_class_weights(train_loader, CONFIG['n_classes'], device)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=CONFIG['label_smoothing']
        )
        logger.info("✓ Using weighted loss with class weights")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("Using standard CrossEntropyLoss")
    
    # Optimizer with different LR for different layers
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,  # Restart every 20 epochs
        T_mult=2,  # Double the period after each restart
        eta_min=1e-6
    )
    
    # Mixed precision
    scaler = GradScaler() if CONFIG['mixed_precision'] and torch.cuda.is_available() else None
    if scaler:
        logger.info("Using mixed precision training")
    
    # Training loop
    best_val_iou = 0.0
    best_active_classes = 0
    history = {
        'train_loss': [],
        'train_iou': [],
        'val_loss': [],
        'val_iou': [],
        'active_classes': [],
        'lr': []
    }
    
    logger.info("="*80)
    logger.info("STARTING TRAINING WITH CLASS WEIGHTS")
    logger.info("="*80)
    
    for epoch in range(CONFIG['num_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        logger.info("-" * 80)
        
        # Train
        train_loss, train_iou, train_per_class, active_classes = train_epoch_with_class_iou(
            model, train_loader, criterion, optimizer, device, CONFIG['n_classes'], scaler
        )
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        logger.info(f"Active classes (IoU > 0.01): {active_classes}/{CONFIG['n_classes']}")
        
        # Show top and bottom performing classes
        sorted_classes = sorted(train_per_class.items(), key=lambda x: x[1], reverse=True)
        logger.info("Top 5 classes:")
        for cls, iou in sorted_classes[:5]:
            logger.info(f"  Class {cls}: {iou:.4f}")
        
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
        history['active_classes'].append(active_classes)
        history['lr'].append(current_lr)
        
        # Save best model (prioritize more active classes over just IoU)
        improved = (val_iou > best_val_iou) or \
                   (val_iou > best_val_iou * 0.95 and active_classes > best_active_classes)
        
        if improved:
            best_val_iou = val_iou
            best_active_classes = active_classes
            best_model_path = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'active_classes': active_classes,
                'config': CONFIG
            }, best_model_path)
            logger.info(f"✓ Saved best model (IoU: {val_iou:.4f}, Active: {active_classes})")
        
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
                'active_classes': active_classes,
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
    logger.info(f"Best Active Classes: {best_active_classes}/{CONFIG['n_classes']}")
    logger.info(f"Models saved to: {checkpoint_dir}")
    logger.info(f"Training history saved to: {history_path}")
    
    logger.info("\nTo test the model, run:")
    logger.info(f"  python test_inference.py")
    logger.info("  (Update CHECKPOINT_PATH to point to the best_model.pth in checkpoints_fixed/)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\n\nTraining failed: {e}", exc_info=True)
