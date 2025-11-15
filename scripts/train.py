#!/usr/bin/env python3
"""
Training Script for ViT-Small Floor Plan Segmentation
Addresses severe class imbalance with weighted loss function
Supports resuming from checkpoints
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
# import torch.nn as nn
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
# import mlflow.pytorch

# Import project modules
from src.data.dataset import create_dataloaders
from src.utils.logging_config import setup_logging
from src.utils.focal_loss import FocalLoss #, create_focal_loss
# Use hybrid model instead of pure ViT
from models.vit_segmentation_hybrid import HybridViTCNNSegmentation, validate # calculate_iou, train_epoch,

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


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """
    Load checkpoint and restore training state
    
    Returns:
        start_epoch: Epoch to resume from
        history: Training history
        best_val_iou: Best validation IoU so far
        best_active_classes: Best active classes count
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("[OK] Model state loaded")
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    logger.info("[OK] Optimizer state loaded")
    
    # Load scheduler state
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    logger.info("[OK] Scheduler state loaded")
    
    # Get training progress
    start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    history = checkpoint.get('history', {
        'train_loss': [],
        'train_iou': [],
        'val_loss': [],
        'val_iou': [],
        'active_classes': [],
        'lr': []
    })
    
    # Get best metrics
    best_val_iou = max(history.get('val_iou', [0.0]))
    best_active_classes = max(history.get('active_classes', [0]))
    
    logger.info(f"[OK] Resuming from epoch {start_epoch}")
    logger.info(f"Previous best IoU: {best_val_iou:.4f}")
    logger.info(f"Previous best active classes: {best_active_classes}")
    
    return start_epoch, history, best_val_iou, best_active_classes


def train_epoch_with_class_iou(model, dataloader, criterion, optimizer, device, n_classes, config, scaler=None):
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
            
            # NEW: Gradient clipping for stability
            if config['gradient_clip'] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            
            # NEW: Gradient clipping
            if config['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            
            optimizer.step()
        
        # NEW: Log metrics every N batches
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


def main(resume_checkpoint=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train ViT Floor Plan Segmentation with class weighting')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from (e.g., models/checkpoints/checkpoint_epoch_10.pth)')
    parser.add_argument('--fresh', action='store_true', default=False,
                       help='Start training from scratch, ignoring any existing checkpoints')
    args = parser.parse_args()
    
    # Determine resume path with hybrid logic (Option 3)
    resume_from = None
    
    if args.resume:
        # Explicit resume path provided
        resume_from = args.resume
        logger.info("=" * 80)
        logger.info("EXPLICIT RESUME: Using provided checkpoint")
        logger.info("=" * 80)
    elif not args.fresh:
        # Auto-resume from best model if it exists (NEW: Option 3)
        best_model_path = Path(resume_checkpoint or 'models/checkpoints/best_model.pth')
        if best_model_path.exists():
            resume_from = str(best_model_path)
            logger.info("=" * 80)
            logger.info("AUTO-RESUME: Loading best_model.pth")
            logger.info("=" * 80)
        else:
            logger.info("=" * 80)
            logger.info("NEW TRAINING: No existing best model found")
            logger.info("=" * 80)
    else:
        # Fresh start requested
        logger.info("=" * 80)
        logger.info("FRESH START: Starting training from scratch (--fresh flag)")
        logger.info("=" * 80)
    
    # Configuration with OPTIMIZED hyperparameters for minority class learning
    CONFIG = {
        # Data
        'images_dir': 'data/processed/images',
        'masks_dir': 'data/processed/annotations',
        'batch_size': 4,  #  Reduced from 12 for better minority class sampling
        'num_workers': 0,
        
        # Model
        'img_size': 512,
        'patch_size': 32,
        'n_classes': 12,
        'embed_dim': 384,
        'n_encoder_layers': 12,
        'n_decoder_layers': 3,
        'n_heads': 6,
        'mlp_ratio': 4.0,
        'dropout': 0.25,  # Increased from 0.15 to prevent overfitting to background
        
        # Training - FIXED for better convergence
        'num_epochs': 50,  # Increased from 40 for more training time
        'learning_rate': 1.5e-4,  # Kept: Working well
        'weight_decay': 0.005,  # Kept: Stable
        'warmup_steps': 1000,  # Warmup for stable training start
        'warmup_epochs': 3,  # Increased from 2 for more stable start
        'gradient_clip': 1.0,  # Kept: Gradient clipping for stability
        'mixed_precision': True,
        
        # Loss function -  for minority classes
        'use_class_weights': True,
        'label_smoothing': 0.05,  # Kept: Sharper predictions
        'focal_loss_alpha': 0.25,  # Kept: Balance foreground/background
        'focal_loss_gamma': 3.0,  # Increased from 2.5 to focus harder on difficult examples
        
        # Checkpointing & Monitoring
        'checkpoint_dir': 'models/checkpoints',
        'save_frequency': 10,
        'log_frequency': 50,  # NEW: Log metrics every N batches
        'early_stopping_patience': 15,  # NEW: Early stopping if no improvement
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
    
    # Create hybrid ViT-CNN model (FIXED: Better for pixel-level segmentation)
    logger.info("Creating Hybrid ViT-CNN Segmentation Model...")
    logger.info("  - ViT Encoder: Global context with 12 layers")
    logger.info("  - CNN Decoder: Progressive upsampling with skip connections")
    logger.info("  - Patch Size: 16px (FIXED: smaller for better spatial details)")
    logger.info("  - Skip Connections: Enabled (FIXED: preserves spatial information)")
    model = HybridViTCNNSegmentation(
        img_size=CONFIG['img_size'],
        patch_size=16,  # FIXED: Smaller patches for finer details (was 32)
        in_channels=3,
        n_classes=CONFIG['n_classes'],
        embed_dim=CONFIG['embed_dim'],
        n_encoder_layers=CONFIG['n_encoder_layers'],
        n_heads=CONFIG['n_heads'],
        mlp_ratio=CONFIG['mlp_ratio'],
        dropout=CONFIG['dropout'],
        skip_connections=True  # Enable skip connections
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params / 1e6:.2f}M")
    logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # Calculate class weights
    if CONFIG['use_class_weights']:
        class_weights = calculate_class_weights(train_loader, CONFIG['n_classes'], device)
        # Use Focal Loss with class weights for extreme imbalance
        criterion = FocalLoss(
            alpha=0.25,
            gamma=2.0,
            reduction='mean',
            class_weights=class_weights
        )
        logger.info("[OK] Using Focal Loss with class weights (alpha=0.25, gamma=2.0)")
        logger.info(f"   Class weights: min={class_weights.min():.4f}, max={class_weights.max():.4f}")
    else:
        # Use Focal Loss without weights if class weighting is disabled
        criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        logger.info("Using Focal Loss (alpha=0.25, gamma=2.0) without class weights")
    
    # Optimizer with different LR for different layers
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # NEW: Warmup scheduler for stable training start
    def warmup_scheduler(epoch):
        warmup_epochs = CONFIG['warmup_epochs']
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 1.0
    
    warmup_sched = optim.lr_scheduler.LambdaLR(optimizer, warmup_scheduler)
    
    # Cosine annealing with warm restarts (applied after warmup)
    # FIXED: Prevent learning rate from decaying too aggressively
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,        # Increased from 30 to 50 - slower decay to prevent regression
        T_mult=1,      # Keep restart period constant (was 2 which grows: 15→30→60)
        eta_min=1e-5   # Higher minimum LR (was 1e-6) - keeps learning
    )
    # Effect: Learning rate stays high enough for model to continue improving
    # throughout all 50 epochs without regression at epoch 20
    # Schedule: Decays over 50 epochs, restarts every 50 epochs at higher LR
    # This prevents the LR collapse that caused validation IoU regression at epoch 20
    
    # Mixed precision
    scaler = GradScaler() if CONFIG['mixed_precision'] and torch.cuda.is_available() else None
    if scaler:
        logger.info("Using mixed precision training")
    
    # Load checkpoint if resuming
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
    else:
        logger.info("="*80)
        logger.info("STARTING TRAINING WITH CLASS WEIGHTS")
        logger.info("="*80)
        if resume_from:
            logger.warning(f"Checkpoint not found: {resume_from}")
            logger.warning("Starting training from scratch")
    
    # MLflow setup with backend verification
    logger.info("=" * 80)
    logger.info("MLFLOW SETUP")
    logger.info("=" * 80)
    
    # Check MLflow tracking URI
    try:
        mlflow_tracking_uri = mlflow.tracking.get_tracking_uri()
        logger.info(f"MLflow tracking URI: {mlflow_tracking_uri}")
    except Exception as e:
        logger.info(f"MLflow tracking: Using default backend ({str(e)[:50]}...)")
    
    # Set experiment
    mlflow.set_experiment("floor-plan-segmentation")
    run_name = f"vit-{CONFIG['n_classes']}classes-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(CONFIG)
    logger.info(f"[MLflow] Experiment started: {run_name}")
    logger.info(f"[MLflow] View dashboard: mlflow ui")
    logger.info(f"[MLflow] Backend location: ./mlruns/")
    logger.info("=" * 80)
    
    for epoch in range(start_epoch, CONFIG['num_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        logger.info("-" * 80)
        
        # Track epoch timing
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_iou, train_per_class, active_classes = train_epoch_with_class_iou(
            model, train_loader, criterion, optimizer, device, CONFIG['n_classes'], CONFIG, scaler
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
        if epoch < CONFIG['warmup_epochs']:
            warmup_sched.step()
        else:
            cosine_scheduler.step(epoch - CONFIG['warmup_epochs'])
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Learning Rate: {current_lr:.6f}")
        
        # Calculate epoch timing
        epoch_time = time.time() - epoch_start_time
        
        # NEW: Enhanced MLflow tracking with per-class metrics
        mlflow_metrics = {
            'train_loss': train_loss,
            'train_iou': train_iou,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'active_classes': active_classes,
            'learning_rate': current_lr,
            'epoch_time_sec': epoch_time,  # NEW: Epoch timing
        }
        
        # Add per-class IoU metrics (NEW: Per-class tracking)
        for cls in range(CONFIG['n_classes']):
            mlflow_metrics[f'train_iou_class_{cls}'] = train_per_class.get(cls, 0.0)
        
        # Log all metrics to MLflow
        mlflow.log_metrics(mlflow_metrics, step=epoch)
        
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
                'scheduler_state_dict': warmup_sched.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'active_classes': active_classes,
                'config': CONFIG
            }, best_model_path)
            logger.info(f"[OK] Saved best model (IoU: {val_iou:.4f}, Active: {active_classes})")
        
        # Save checkpoint periodically
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
            
            # NEW: Log checkpoint as MLflow artifact
            mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
            logger.info(f"[MLflow] Logged checkpoint artifact: {checkpoint_path.name}")
    
    # Save final model and history
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
    
    # Log final artifacts to MLflow
    mlflow.log_artifact(str(best_model_path), artifact_path="models")
    mlflow.log_artifact(str(final_model_path), artifact_path="models")
    mlflow.log_artifact(str(history_path), artifact_path="metrics")
    mlflow.log_artifact(str(config_path), artifact_path="config")
    logger.info("[MLflow] Logged all final artifacts")
    
    # Log final metrics and summary
    final_metrics = {
        'final_best_val_iou': best_val_iou,
        'final_best_active_classes': best_active_classes,
        'final_epoch': CONFIG['num_epochs'] - 1,
        'total_epochs_trained': len(history['train_loss']),
    }
    mlflow.log_metrics(final_metrics)
    logger.info("[MLflow] Logged final metrics summary")
    
    # Log tags for easy filtering in MLflow UI
    mlflow.set_tags({
        'model_type': 'ViT-Small-Segmentation',
        'dataset': 'CubiCasa5K',
        'num_classes': CONFIG['n_classes'],
        'focal_loss_enabled': 'true',
        'class_weights_enabled': str(CONFIG['use_class_weights']),
    })
    logger.info("[MLflow] Set tracking tags")
    
    mlflow.end_run()
    logger.info("[MLflow] Run completed")
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"Best Val IoU: {best_val_iou:.4f}")
    logger.info(f"Best Active Classes: {best_active_classes}/{CONFIG['n_classes']}")
    logger.info(f"Models saved to: {checkpoint_dir}")
    logger.info(f"Training history saved to: {history_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("NEXT STEPS")
    logger.info("=" * 80)
    
    logger.info("\n[Continue Training]")
    logger.info("  python scripts/train.py")
    logger.info("  (Auto-resumes from best_model.pth)")
    
    logger.info("\n[Start Fresh (Reset Progress)]")
    logger.info("  python scripts/train.py --fresh")
    logger.info("  (Ignores existing best_model.pth)")
    
    logger.info("\n[Resume from Specific Checkpoint]")
    logger.info("  python scripts/train.py --resume models/checkpoints/checkpoint_epoch_30.pth")
    
    logger.info("\n[View MLflow Dashboard]")
    logger.info("  mlflow ui")
    logger.info("  Open: http://localhost:5000")
    
    logger.info("\n[Test the Model]")
    logger.info("  python scripts/inference.py --model models/checkpoints/best_model.pth")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\n\nTraining failed: {e}", exc_info=True)
