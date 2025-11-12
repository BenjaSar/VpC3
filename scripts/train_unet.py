#!/usr/bin/env python3
"""
Training Script for UNet++ Floor Plan Segmentation
Uses EfficientNet-B0 encoder with proven architecture
Expected convergence: 15-20 epochs to 0.50+ IoU
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import json
import argparse
from datetime import datetime
from collections import Counter
import time
import mlflow
import mlflow.pytorch

# Import project modules
from src.data.dataset import create_dataloaders
from src.utils.logging_config import setup_logging
from src.utils.focal_loss import FocalLoss
from models.unet_plusplus_segmentation import UNetPlusPlusSegmentation, calculate_iou, train_epoch, validate

logger = setup_logging()


def calculate_class_weights(dataloader, num_classes=12, device='cpu'):
    """Calculate class weights based on inverse frequency"""
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
        count = class_counts.get(i, 1)
        weight = total_pixels / (num_classes * count)
        weight = min(weight, 100.0)
        weights.append(weight)
    
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * num_classes
    
    logger.info("\nClass Weight Statistics:")
    logger.info(f"  Min weight: {weights.min():.4f}")
    logger.info(f"  Max weight: {weights.max():.4f}")
    logger.info(f"  Mean weight: {weights.mean():.4f}")
    
    return weights.to(device)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load checkpoint and restore training state"""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("[OK] Model state loaded")
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    logger.info("[OK] Optimizer state loaded")
    
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    logger.info("[OK] Scheduler state loaded")
    
    start_epoch = checkpoint['epoch'] + 1
    history = checkpoint.get('history', {
        'train_loss': [],
        'train_iou': [],
        'val_loss': [],
        'val_iou': [],
        'active_classes': [],
        'lr': []
    })
    
    best_val_iou = max(history.get('val_iou', [0.0]))
    best_active_classes = max(history.get('active_classes', [0]))
    
    logger.info(f"[OK] Resuming from epoch {start_epoch}")
    logger.info(f"Previous best IoU: {best_val_iou:.4f}")
    
    return start_epoch, history, best_val_iou, best_active_classes


def train_epoch_with_class_iou(model, dataloader, criterion, optimizer, device, n_classes, config, scaler=None):
    """Train for one epoch with per-class IoU tracking"""
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
            if config['gradient_clip'] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            if config['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            optimizer.step()
        
        if batch_idx % config['log_frequency'] == 0 and batch_idx > 0:
            logger.info(f"  Batch {batch_idx}/{len(dataloader)}: Loss={loss.item():.4f}")
        
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
        
        mean_iou = np.mean([np.mean(ious) if ious else 0 for ious in class_ious.values()])
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mIoU': f'{mean_iou:.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    per_class_iou = {cls: np.mean(ious) if ious else 0.0 
                     for cls, ious in class_ious.items()}
    mean_iou = np.mean(list(per_class_iou.values()))
    active_classes = sum(1 for iou in per_class_iou.values() if iou > 0.01)
    
    return avg_loss, mean_iou, per_class_iou, active_classes


def main():
    parser = argparse.ArgumentParser(description='Train UNet++ for Floor Plan Segmentation')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--fresh', action='store_true', default=False,
                       help='Start training from scratch')
    args = parser.parse_args()
    
    resume_from = None
    if args.resume:
        resume_from = args.resume
        logger.info("=" * 80)
        logger.info("EXPLICIT RESUME: Using provided checkpoint")
        logger.info("=" * 80)
    elif not args.fresh:
        best_model_path = Path('models/checkpoints/best_model.pth')
        if best_model_path.exists():
            resume_from = str(best_model_path)
            logger.info("=" * 80)
            logger.info("AUTO-RESUME: Loading best_model.pth")
            logger.info("=" * 80)
    else:
        logger.info("=" * 80)
        logger.info("FRESH START: Starting training from scratch (--fresh flag)")
        logger.info("=" * 80)
    
    CONFIG = {
        'images_dir': 'data/processed/images',
        'masks_dir': 'data/processed/annotations',
        'batch_size': 4,
        'num_workers': 0,
        'img_size': 512,
        'n_classes': 12,
        'num_epochs': 50,
        'learning_rate': 1.5e-4,
        'weight_decay': 0.005,
        'warmup_epochs': 3,
        'gradient_clip': 1.0,
        'mixed_precision': True,
        'use_class_weights': True,
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 3.0,
        'checkpoint_dir': 'models/checkpoints',
        'save_frequency': 10,
        'log_frequency': 50,
    }
    
    checkpoint_dir = Path(CONFIG['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = checkpoint_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=4)
    logger.info(f"Configuration saved to {config_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
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
    
    logger.info("Creating UNet++ Model...")
    logger.info("  - Encoder: EfficientNet-B0 (pretrained ImageNet)")
    logger.info("  - Decoder: Dense skip connections")
    logger.info("  - Expected: 15-20 epochs to 0.50+ IoU")
    
    model = UNetPlusPlusSegmentation(
        in_channels=3,
        num_classes=CONFIG['n_classes']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params / 1e6:.2f}M")
    logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    if CONFIG['use_class_weights']:
        class_weights = calculate_class_weights(train_loader, CONFIG['n_classes'], device)
        criterion = FocalLoss(
            alpha=CONFIG['focal_loss_alpha'],
            gamma=CONFIG['focal_loss_gamma'],
            reduction='mean',
            class_weights=class_weights
        )
        logger.info("[OK] Using Focal Loss with class weights")
    else:
        criterion = FocalLoss(alpha=CONFIG['focal_loss_alpha'], 
                             gamma=CONFIG['focal_loss_gamma'], reduction='mean')
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    def warmup_scheduler(epoch):
        warmup_epochs = CONFIG['warmup_epochs']
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 1.0
    
    warmup_sched = optim.lr_scheduler.LambdaLR(optimizer, warmup_scheduler)
    
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,
        T_mult=1,
        eta_min=1e-5
    )
    
    scaler = GradScaler() if CONFIG['mixed_precision'] and torch.cuda.is_available() else None
    if scaler:
        logger.info("Using mixed precision training")
    
    start_epoch = 0
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
    
    if resume_from and Path(resume_from).exists():
        logger.info("="*80)
        logger.info("RESUMING TRAINING FROM CHECKPOINT")
        logger.info("="*80)
        start_epoch, history, best_val_iou, best_active_classes = load_checkpoint(
            resume_from, model, optimizer, warmup_sched, device
        )
    
    mlflow.set_experiment("floor-plan-segmentation")
    run_name = f"unet++-{CONFIG['n_classes']}classes-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(CONFIG)
    logger.info(f"[MLflow] Experiment started: {run_name}")
    
    for epoch in range(start_epoch, CONFIG['num_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        logger.info("-" * 80)
        
        epoch_start_time = time.time()
        
        train_loss, train_iou, train_per_class, active_classes = train_epoch_with_class_iou(
            model, train_loader, criterion, optimizer, device, CONFIG['n_classes'], CONFIG, scaler
        )
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        logger.info(f"Active classes (IoU > 0.01): {active_classes}/{CONFIG['n_classes']}")
        
        sorted_classes = sorted(train_per_class.items(), key=lambda x: x[1], reverse=True)
        logger.info("Top 5 classes:")
        for cls, iou in sorted_classes[:5]:
            logger.info(f"  Class {cls}: {iou:.4f}")
        
        val_loss, val_iou = validate(model, val_loader, criterion, device, CONFIG['n_classes'])
        logger.info(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        
        if epoch < CONFIG['warmup_epochs']:
            warmup_sched.step()
        else:
            cosine_scheduler.step(epoch - CONFIG['warmup_epochs'])
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Learning Rate: {current_lr:.6f}")
        
        epoch_time = time.time() - epoch_start_time
        
        mlflow_metrics = {
            'train_loss': train_loss,
            'train_iou': train_iou,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'active_classes': active_classes,
            'learning_rate': current_lr,
            'epoch_time_sec': epoch_time,
        }
        
        mlflow.log_metrics(mlflow_metrics, step=epoch)
        
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['active_classes'].append(active_classes)
        history['lr'].append(current_lr)
        
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
                'scheduler_state_dict': warmup_sched.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'active_classes': active_classes,
                'history': history,
                'config': CONFIG
            }, best_model_path)
            logger.info(f"[OK] Saved best model (IoU: {val_iou:.4f}, Active: {active_classes})")
        
        if (epoch + 1) % CONFIG['save_frequency'] == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': warmup_sched.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'active_classes': active_classes,
                'history': history,
                'config': CONFIG
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
    
    final_model_path = checkpoint_dir / 'final_model.pth'
    torch.save({
        'epoch': CONFIG['num_epochs'] - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': warmup_sched.state_dict(),
        'history': history,
        'config': CONFIG
    }, final_model_path)
    
    history_path = checkpoint_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    mlflow.log_artifact(str(best_model_path), artifact_path="models")
    mlflow.log_artifact(str(final_model_path), artifact_path="models")
    mlflow.log_artifact(str(history_path), artifact_path="metrics")
    
    mlflow.end_run()
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"Best Val IoU: {best_val_iou:.4f}")
    logger.info(f"Best Active Classes: {best_active_classes}/{CONFIG['n_classes']}")
    logger.info(f"Models saved to: {checkpoint_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\n\nTraining failed: {e}", exc_info=True)
