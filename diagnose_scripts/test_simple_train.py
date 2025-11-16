#!/usr/bin/env python3
"""
Diagnostic Script: Simple Training Test
Tests training WITHOUT class weights to isolate the issue
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
import numpy as np
from tqdm import tqdm

from src.data.dataset import create_dataloaders
from src.utils.logging_config import setup_logging
from models.vit_segmentation import ViTSegmentation, train_epoch, validate

logger = setup_logging()

def test_simple_training():
    """Test training with simplified configuration"""
    
    print("=" * 80)
    print("SIMPLE TRAINING TEST")
    print("=" * 80)
    
    # Simplified configuration (NO class weights)
    CONFIG = {
        'images_dir': 'data/processed/images',
        'masks_dir': 'data/processed/annotations',
        'batch_size': 4,
        'num_workers': 0,
        'img_size': 512,
        'patch_size': 32,
        'n_classes': 12,
        'embed_dim': 384,
        'n_encoder_layers': 12,
        'n_decoder_layers': 3,
        'n_heads': 6,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'num_epochs': 3,  # Just 3 epochs for testing
        'learning_rate': 1e-4,  # Higher LR than original
        'weight_decay': 0.01,
        'use_class_weights': False,  # NO class weights!
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    try:
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
    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}")
        import traceback
        traceback.print_exc()
        return
    
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
    
    # Loss and optimizer (NO class weights)
    criterion = nn.CrossEntropyLoss()  # Standard loss
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    
    logger.info("=" * 80)
    logger.info("TRAINING WITHOUT CLASS WEIGHTS")
    logger.info("=" * 80)
    
    # Training loop
    for epoch in range(CONFIG['num_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        logger.info("-" * 80)
        
        # Train
        train_loss, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, device, CONFIG['n_classes']
        )
        logger.info(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        
        # Validate
        val_loss, val_iou = validate(model, val_loader, criterion, device, CONFIG['n_classes'])
        logger.info(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        
        # Scheduler step
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        logger.info(f"Learning Rate: {lr:.6f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETED")
    logger.info("=" * 80)
    logger.info("\nIf loss decreased and IoU increased:")
    logger.info("  → Problem is with class weights or training configuration")
    logger.info("  → Try different class weight strategy")
    logger.info("\nIf loss didn't improve:")
    logger.info("  → Problem might be with dataset or learning rate")
    logger.info("  → Run diagnose_dataset.py to check data")

if __name__ == "__main__":
    try:
        test_simple_training()
    except KeyboardInterrupt:
        logger.info("\n\nTest interrupted by user")
    except Exception as e:
        logger.error(f"\n\nTest failed: {e}", exc_info=True)
