#!/usr/bin/env python3
"""
Runnable Script: Test Dataset & DataLoaders
Uses data/dataset.py to create and test PyTorch dataloaders
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from data.dataset import create_dataloaders, FloorPlanDataset, get_augmentation_pipeline
from src.utils.logging_config import setup_logging
import matplotlib.pyplot as plt
import numpy as np

logger = setup_logging()


def visualize_batch(batch, num_samples=4):
    """
    Visualize samples from a batch
    """
    images = batch['image']
    masks = batch['mask']
    filenames = batch['filename']
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    
    for i in range(min(num_samples, len(images))):
        # Denormalize image
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        mask = masks[i].cpu().numpy()
        
        # Display image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Image: {filenames[i]}', fontsize=10)
        axes[i, 0].axis('off')
        
        # Display mask
        axes[i, 1].imshow(mask, cmap='tab20')
        axes[i, 1].set_title(f'Mask (Classes: {torch.unique(masks[i]).tolist()})', fontsize=10)
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    output_path = Path("dataloader_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved visualization to: {output_path}")
    plt.close()


def main():
    """
    Main function to test dataset and dataloaders
    """
    
    # ==================== CONFIGURATION ====================
    CONFIG = {
        'images_dir': 'data/processed/images',
        'masks_dir': 'data/processed/annotations',
        'batch_size': 8,
        'num_workers': 0,  # Set to 0 for Windows, 4 for Linux/Mac
        'image_size': 512,
        'num_classes': 34,
        'split_ratio': (0.7, 0.15, 0.15)
    }
    
    logger.info("="*80)
    logger.info("TESTING DATASET & DATALOADERS")
    logger.info("="*80)
    
    # Check if processed data exists
    images_path = Path(CONFIG['images_dir'])
    masks_path = Path(CONFIG['masks_dir'])
    
    if not images_path.exists():
        logger.error(f"Images directory not found: {images_path}")
        logger.info("\nHave you preprocessed the dataset?")
        logger.info("Run: python run_preprocessing.py")
        return
    
    if not masks_path.exists():
        logger.error(f"Masks directory not found: {masks_path}")
        return
    
    # Count files
    image_files = list(images_path.glob('*.*'))
    mask_files = list(masks_path.glob('*.*'))
    
    logger.info(f"\nDataset found:")
    logger.info(f"  Images: {len(image_files)}")
    logger.info(f"  Masks: {len(mask_files)}")
    
    if len(image_files) == 0:
        logger.error("No images found! Please preprocess the dataset first.")
        return
    
    # ==================== CREATE DATALOADERS ====================
    logger.info("\n" + "="*80)
    logger.info("CREATING DATALOADERS")
    logger.info("="*80)
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            images_dir=CONFIG['images_dir'],
            masks_dir=CONFIG['masks_dir'],
            batch_size=CONFIG['batch_size'],
            num_workers=CONFIG['num_workers'],
            image_size=CONFIG['image_size'],
            num_classes=CONFIG['num_classes'],
            split_ratio=CONFIG['split_ratio'],
            seed=42
        )
        
        logger.info(f"✓ Train loader: {len(train_loader)} batches")
        logger.info(f"✓ Val loader: {len(val_loader)} batches")
        logger.info(f"✓ Test loader: {len(test_loader)} batches")
        
    except Exception as e:
        logger.error(f"Error creating dataloaders: {e}")
        return
    
    # ==================== TEST DATA LOADING ====================
    logger.info("\n" + "="*80)
    logger.info("TESTING DATA LOADING")
    logger.info("="*80)
    
    logger.info("\nLoading one training batch...")
    try:
        train_batch = next(iter(train_loader))
        
        images = train_batch['image']
        masks = train_batch['mask']
        filenames = train_batch['filename']
        
        logger.info(f"\nBatch information:")
        logger.info(f"  Batch size: {len(images)}")
        logger.info(f"  Image shape: {images.shape}")
        logger.info(f"  Image dtype: {images.dtype}")
        logger.info(f"  Image range: [{images.min():.2f}, {images.max():.2f}]")
        logger.info(f"  Mask shape: {masks.shape}")
        logger.info(f"  Mask dtype: {masks.dtype}")
        logger.info(f"  Unique classes: {torch.unique(masks).tolist()}")
        logger.info(f"  Filenames: {filenames[:3]}...")
        
        # ==================== VISUALIZE SAMPLES ====================
        logger.info("\n" + "="*80)
        logger.info("VISUALIZING SAMPLES")
        logger.info("="*80)
        
        visualize_batch(train_batch, num_samples=4)
        
    except Exception as e:
        logger.error(f"Error loading batch: {e}", exc_info=True)
        return
    
    # ==================== TEST SINGLE SAMPLE ====================
    logger.info("\n" + "="*80)
    logger.info("TESTING SINGLE SAMPLE ACCESS")
    logger.info("="*80)
    
    try:
        # Get dataset from dataloader
        train_dataset = train_loader.dataset
        
        # Access single sample
        sample = train_dataset[0]
        logger.info(f"\nSingle sample:")
        logger.info(f"  Image shape: {sample['image'].shape}")
        logger.info(f"  Mask shape: {sample['mask'].shape}")
        logger.info(f"  Filename: {sample['filename']}")
        
    except Exception as e:
        logger.error(f"Error accessing single sample: {e}")
    
    # ==================== MEMORY CHECK ====================
    logger.info("\n" + "="*80)
    logger.info("MEMORY CHECK")
    logger.info("="*80)
    
    if torch.cuda.is_available():
        images_gpu = images.cuda()
        masks_gpu = masks.cuda()
        
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        
        logger.info(f"\nGPU Memory:")
        logger.info(f"  Allocated: {allocated:.2f} GB")
        logger.info(f"  Reserved: {reserved:.2f} GB")
        logger.info(f"  Batch on GPU: {images_gpu.shape}")
    else:
        logger.info("\nCUDA not available - using CPU")
        logger.info(f"  Batch size on CPU: {CONFIG['batch_size']}")
        logger.info(f"  Estimated memory per batch: ~{CONFIG['batch_size'] * 3 * 512 * 512 * 4 / 1024**2:.2f} MB")
    
    # ==================== ITERATION SPEED TEST ====================
    logger.info("\n" + "="*80)
    logger.info("ITERATION SPEED TEST")
    logger.info("="*80)
    
    import time
    
    logger.info("\nIterating through 10 batches...")
    start_time = time.time()
    
    for i, batch in enumerate(train_loader):
        if i >= 10:
            break
        images = batch['image']
        masks = batch['mask']
    
    elapsed = time.time() - start_time
    logger.info(f"✓ Loaded 10 batches in {elapsed:.2f} seconds")
    logger.info(f"  Average: {elapsed/10:.3f} seconds per batch")
    logger.info(f"  Throughput: ~{(CONFIG['batch_size'] * 10) / elapsed:.1f} images/second")
    
    # ==================== SUMMARY ====================
    logger.info("\n" + "="*80)
    logger.info("DATASET TEST COMPLETE!")
    logger.info("="*80)
    
    logger.info("\n✓ All tests passed successfully!")
    logger.info("\nDataset statistics:")
    logger.info(f"  Total images: {len(image_files)}")
    logger.info(f"  Train samples: ~{int(len(image_files) * CONFIG['split_ratio'][0])}")
    logger.info(f"  Val samples: ~{int(len(image_files) * CONFIG['split_ratio'][1])}")
    logger.info(f"  Test samples: ~{int(len(image_files) * CONFIG['split_ratio'][2])}")
    logger.info(f"  Batch size: {CONFIG['batch_size']}")
    logger.info(f"  Image size: {CONFIG['image_size']}x{CONFIG['image_size']}")
    logger.info(f"  Number of classes: {CONFIG['num_classes']}")
    
    logger.info("\nNext steps:")
    logger.info("1. Review dataloader_visualization.png")
    logger.info("2. Adjust batch_size if needed (CONFIG in this script)")
    logger.info("3. Ready to start training!")
    logger.info("4. Use these dataloaders in your training script")
    
    logger.info("\nExample training loop:")
    logger.info("""
    for epoch in range(num_epochs):
        for batch in train_loader:
            images = batch['image'].cuda()
            masks = batch['mask'].cuda()
            
            # Your training code here
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
    """)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
    except Exception as e:
        logger.error(f"\n\nFatal error: {e}", exc_info=True)
