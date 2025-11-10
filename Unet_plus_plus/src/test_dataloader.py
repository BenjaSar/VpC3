#!/usr/bin/env python3
"""
TEST DATASET & DATALOADERS
═══════════════════════════════════════════════════════════════════════════

Complete test script for floor plan dataset and PyTorch dataloaders.
Tests data loading, visualization, memory usage, and performance.

Usage:
    python TEST_DATALOADER.py

Expected data structure:
    data/processed/
    ├── images/
    └── annotations/

Author: Data Scientist
Date: 2025-11-06
Version: 1.0
"""

import sys
from pathlib import Path
import time
import warnings
from typing import Dict

import torch
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.dataset import create_dataloaders, FloorPlanDataset
from src.utils.logging_config import setup_logging

logger = setup_logging(log_level=logging.INFO)


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def visualize_batch(batch: Dict, num_samples: int = 4, save_path: Path = None):
    """
    Visualize samples from a batch
    
    Args:
        batch: Batch dictionary from dataloader
        num_samples: Number of samples to visualize
        save_path: Path to save visualization
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not available. Skipping visualization.")
        return
    
    images = batch['image']
    masks = batch['mask']
    
    num_samples = min(num_samples, len(images))
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Get image
        img = images[i].cpu().numpy()
        
        # Denormalize if normalized
        if img.max() <= 1.5:  # Likely normalized
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            if img.shape[0] == 3:  # CHW format
                img = img.transpose(1, 2, 0)
                img = img * std + mean
            else:  # HWC format
                img = img * std + mean
        else:  # Already in 0-255 range
            img = img.transpose(1, 2, 0) if img.shape[0] == 3 else img
        
        img = np.clip(img, 0, 1)
        
        # Get mask
        mask = masks[i].cpu().numpy()
        
        # Display image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Image {i} - Shape: {img.shape}', fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Display mask with colormap
        unique_classes = np.unique(mask)
        im = axes[i, 1].imshow(mask, cmap='tab20', vmin=0, vmax=33)
        axes[i, 1].set_title(
            f'Mask {i} - Classes: {len(unique_classes)} - Values: {unique_classes.tolist()[:5]}...',
            fontsize=10,
            fontweight='bold'
        )
        axes[i, 1].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i, 1], label='Class Index')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = Path('outputs/dataloader_visualization.png')
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Visualization saved to {save_path}")
    plt.close()


def print_batch_info(batch: Dict):
    """Print detailed batch information"""
    images = batch['image']
    masks = batch['mask']
    
    logger.info("\n" + "=" * 80)
    logger.info("BATCH INFORMATION")
    logger.info("=" * 80)
    
    logger.info(f"\n📊 Tensors:")
    logger.info(f"   Images shape:     {images.shape}")
    logger.info(f"   Images dtype:     {images.dtype}")
    logger.info(f"   Images range:     [{images.min():.4f}, {images.max():.4f}]")
    logger.info(f"   Images memory:    {images.element_size() * images.nelement() / 1024**2:.2f} MB")
    
    logger.info(f"\n   Masks shape:      {masks.shape}")
    logger.info(f"   Masks dtype:      {masks.dtype}")
    logger.info(f"   Masks range:      [{masks.min()}, {masks.max()}]")
    logger.info(f"   Masks memory:     {masks.element_size() * masks.nelement() / 1024**2:.2f} MB")
    
    logger.info(f"\n📁 Metadata:")
    logger.info(f"   Batch size:       {len(images)}")
    
    # Unique classes in batch
    unique_classes = torch.unique(masks)
    logger.info(f"   Unique classes:   {len(unique_classes)} - {unique_classes.tolist()}")
    
    # Class distribution in batch
    class_counts = torch.bincount(masks.view(-1), minlength=35)[:34]
    non_zero_classes = (class_counts > 0).sum().item()
    logger.info(f"   Classes in batch: {non_zero_classes}/34")


# ═══════════════════════════════════════════════════════════════════════════
# PERFORMANCE TESTING
# ═══════════════════════════════════════════════════════════════════════════

def test_loading_speed(dataloader, num_batches: int = 10):
    """
    Test data loading speed
    
    Args:
        dataloader: DataLoader to test
        num_batches: Number of batches to load
    """
    logger.info("\n" + "=" * 80)
    logger.info("LOADING SPEED TEST")
    logger.info("=" * 80)
    
    logger.info(f"\nLoading {num_batches} batches...")
    
    times = []
    start_total = time.time()
    
    for i, batch in enumerate(dataloader):
        batch_start = time.time()
        
        # Access tensors to ensure full loading
        images = batch['image']
        masks = batch['mask']
        
        batch_time = time.time() - batch_start
        times.append(batch_time)
        
        if i >= num_batches - 1:
            break
    
    total_time = time.time() - start_total
    
    logger.info(f"\n📈 Results:")
    logger.info(f"   Total time:       {total_time:.2f} seconds")
    logger.info(f"   Average time:     {np.mean(times):.4f} seconds/batch")
    logger.info(f"   Min time:         {np.min(times):.4f} seconds/batch")
    logger.info(f"   Max time:         {np.max(times):.4f} seconds/batch")
    logger.info(f"   Std deviation:    {np.std(times):.4f} seconds")
    
    batch_size = len(batch['image'])
    samples_per_sec = (batch_size * num_batches) / total_time
    logger.info(f"   Throughput:       {samples_per_sec:.1f} samples/second")


def test_gpu_memory(batch: Dict):
    """
    Test GPU memory usage
    
    Args:
        batch: Batch from dataloader
    """
    if not torch.cuda.is_available():
        logger.info("\n⚠️  CUDA not available - skipping GPU memory test")
        return
    
    logger.info("\n" + "=" * 80)
    logger.info("GPU MEMORY TEST")
    logger.info("=" * 80)
    
    images = batch['image']
    masks = batch['mask']
    
    # Clear cache
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Move to GPU
    logger.info("\nMoving batch to GPU...")
    images_gpu = images.cuda()
    masks_gpu = masks.cuda()
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    
    logger.info(f"\n💾 GPU Memory:")
    logger.info(f"   Allocated:        {allocated:.3f} GB")
    logger.info(f"   Reserved:         {reserved:.3f} GB")
    logger.info(f"   Peak allocated:   {max_allocated:.3f} GB")
    
    batch_size = len(images)
    logger.info(f"\n📊 Per-sample:")
    logger.info(f"   Batch size:       {batch_size}")
    logger.info(f"   Memory/sample:    {(allocated / batch_size * 1024):.1f} MB")
    logger.info(f"   Estimated for B=8:    {(allocated / batch_size * 8):.3f} GB")


def test_cpu_memory():
    """Test CPU memory usage"""
    try:
        import psutil
    except ImportError:
        logger.warning("psutil not installed. Skipping CPU memory test.")
        return
    
    logger.info("\n" + "=" * 80)
    logger.info("CPU MEMORY TEST")
    logger.info("=" * 80)
    
    process = psutil.Process()
    mem_info = process.memory_info()
    
    rss = mem_info.rss / 1024**3  # Resident set size
    vms = mem_info.vms / 1024**3  # Virtual memory size
    
    logger.info(f"\n💾 CPU Memory:")
    logger.info(f"   RSS (actual):     {rss:.3f} GB")
    logger.info(f"   VMS (virtual):    {vms:.3f} GB")


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

def validate_batch_data(batch: Dict, num_classes: int = 34):
    """
    Validate batch data integrity
    
    Args:
        batch: Batch from dataloader
        num_classes: Expected number of classes
    """
    logger.info("\n" + "=" * 80)
    logger.info("DATA VALIDATION")
    logger.info("=" * 80)
    
    images = batch['image']
    masks = batch['mask']
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: Image shape
    checks_total += 1
    if len(images.shape) == 4 and images.shape[1] == 3:
        logger.info("✓ Image shape correct (B, C, H, W)")
        checks_passed += 1
    else:
        logger.error(f"✗ Image shape incorrect: {images.shape}")
    
    # Check 2: Mask shape
    checks_total += 1
    if len(masks.shape) == 3:
        logger.info("✓ Mask shape correct (B, H, W)")
        checks_passed += 1
    else:
        logger.error(f"✗ Mask shape incorrect: {masks.shape}")
    
    # Check 3: Image dtype
    checks_total += 1
    if images.dtype == torch.float32:
        logger.info("✓ Image dtype correct (float32)")
        checks_passed += 1
    else:
        logger.error(f"✗ Image dtype incorrect: {images.dtype}")
    
    # Check 4: Mask dtype
    checks_total += 1
    if masks.dtype in [torch.int64, torch.int32, torch.long]:
        logger.info("✓ Mask dtype correct (int64/int32)")
        checks_passed += 1
    else:
        logger.error(f"✗ Mask dtype incorrect: {masks.dtype}")
    
    # Check 5: Image value range
    checks_total += 1
    if images.min() >= -5 and images.max() <= 5:  # Allow for normalized values
        logger.info(f"✓ Image values in reasonable range: [{images.min():.3f}, {images.max():.3f}]")
        checks_passed += 1
    else:
        logger.error(f"✗ Image values out of range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Check 6: Mask value range
    checks_total += 1
    if masks.min() >= 0 and masks.max() < num_classes:
        logger.info(f"✓ Mask values in valid range: [{masks.min()}, {masks.max()}] < {num_classes}")
        checks_passed += 1
    else:
        logger.error(f"✗ Mask values out of range: [{masks.min()}, {masks.max()}]")
    
    # Check 7: Size consistency
    checks_total += 1
    if images.shape[2:] == masks.shape[1:]:
        logger.info(f"✓ Image and mask spatial dimensions match: {images.shape[2:]}")
        checks_passed += 1
    else:
        logger.error(f"✗ Spatial dimensions mismatch: image {images.shape[2:]} vs mask {masks.shape[1:]}")
    
    # Check 8: Batch size consistency
    checks_total += 1
    if images.shape[0] == masks.shape[0]:
        logger.info(f"✓ Batch size consistent: {images.shape[0]}")
        checks_passed += 1
    else:
        logger.error(f"✗ Batch size mismatch: image {images.shape[0]} vs mask {masks.shape[0]}")
    
    logger.info(f"\n📋 Validation Result: {checks_passed}/{checks_total} checks passed")
    
    return checks_passed == checks_total


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TEST FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main test function"""
    
    # ==================== CONFIGURATION ====================
    config = {
        'data_root': '/home/ubuntu/floorplan_classifier/VpC3/Unet_plus_plus/data/processed',
        'batch_size': 4,
        'num_workers': 4,  # Set to 0 on Windows if issues
        'image_size': 512,
        'num_classes': 34,
    }
    
    print("\n" + "=" * 80)
    print("FLOOR PLAN DATALOADER TEST")
    print("=" * 80)
    
    # ==================== STEP 1: VERIFY DATA ====================
    logger.info("\n📂 Step 1: Verifying data structure...")
    
    data_root = Path(config['data_root'])
    if not data_root.exists():
        logger.error(f"❌ Data directory not found: {data_root}")
        logger.info("\nExpected structure:")
        logger.info("  data/processed/")
        logger.info("  ├── images/")
        logger.info("  └── annotations/")
        return False
    
    logger.info(f"✓ Data directory found: {data_root}")
    
    # ==================== STEP 2: CREATE DATALOADERS ====================
    logger.info("\n🔄 Step 2: Creating dataloaders...")
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            images_dir='data/processed/images',
            masks_dir='data/processed/annotations',
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            image_size=config['image_size'],
            num_classes=config['num_classes'],
        )
        logger.info("✓ Dataloaders created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create dataloaders: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ==================== STEP 3: LOAD BATCH ====================
    logger.info("\n📥 Step 3: Loading training batch...")
    
    try:
        batch = next(iter(train_loader))
        logger.info("✓ Batch loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load batch: {e}")
        return False
    
    # ==================== STEP 4: VALIDATE DATA ====================
    logger.info("\n✅ Step 4: Validating batch data...")
    
    if not validate_batch_data(batch, config['num_classes']):
        logger.error("❌ Data validation failed")
        return False
    
    # ==================== STEP 5: PRINT BATCH INFO ====================
    logger.info("\n📊 Step 5: Batch statistics...")
    print_batch_info(batch)
    
    # ==================== STEP 6: VISUALIZE ====================
    logger.info("\n🎨 Step 6: Visualizing samples...")
    
    try:
        visualize_batch(batch, num_samples=4, save_path='outputs/batch_visualization.png')
        logger.info("✓ Visualization created")
    except Exception as e:
        logger.warning(f"⚠️  Visualization failed: {e}")
    
    # ==================== STEP 7: PERFORMANCE TESTS ====================
    logger.info("\n⚡ Step 7: Performance testing...")
    
    test_loading_speed(train_loader, num_batches=10)
    
    # ==================== STEP 8: MEMORY TESTS ====================
    logger.info("\n💾 Step 8: Memory testing...")
    
    test_gpu_memory(batch)
    test_cpu_memory()
    
    # ==================== SUMMARY ====================
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    logger.info("\n✅ ALL TESTS PASSED!\n")
    
    logger.info("📊 Dataloader Configuration:")
    logger.info(f"   Data root:        {config['data_root']}")
    logger.info(f"   Batch size:       {config['batch_size']}")
    logger.info(f"   Workers:          {config['num_workers']}")
    logger.info(f"   Image size:       {config['image_size']}x{config['image_size']}")
    
    logger.info(f"\n📈 Dataset Split:")
    logger.info(f"   Train batches:    {len(train_loader)}")
    logger.info(f"   Val batches:      {len(val_loader)}")
    logger.info(f"   Test batches:     {len(test_loader)}")
    
    logger.info("\n✨ Ready for training!")
    logger.info("\nNext steps:")
    logger.info("1. Import dataloaders in your training script")
    logger.info("2. Use the dataloaders in your training loop")
    logger.info("3. Check documentation for examples")
    
    return True


if __name__ == '__main__':
    try:
        success = main()
        
        if success:
            print("\n" + "=" * 80)
            print("✅ DATALOADER TEST COMPLETE - READY TO USE")
            print("=" * 80 + "\n")
            sys.exit(0)
        else:
            print("\n" + "=" * 80)
            print("❌ DATALOADER TEST FAILED")
            print("=" * 80 + "\n")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"\n\n❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
