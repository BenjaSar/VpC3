#!/home/ubuntu/floorplan_classifier/VpC3/floorplan_vit/bin/python3

"""
Comprehensive Data Quality Diagnostic Script
Checks for data alignment, class distribution, and potential issues
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import json
from data.dataset import create_dataloaders
from src.utils.logging_config import setup_logging

logger = setup_logging()


def diagnose_data_quality():
    """Run comprehensive data quality checks"""
    
    logger.info("="*80)
    logger.info("DATA QUALITY DIAGNOSTIC REPORT")
    logger.info("="*80)
    
    # Configuration
    CONFIG = {
        'images_dir': 'data/processed/images',
        'masks_dir': 'data/processed/annotations',
        'batch_size': 1,
        'num_workers': 0,
        'image_size': 512,
        'num_classes': 256
    }
    
    # Check if directories exist
    logger.info("\n1. CHECKING DIRECTORIES")
    logger.info("-" * 80)
    images_dir = Path(CONFIG['images_dir'])
    masks_dir = Path(CONFIG['masks_dir'])
    
    if not images_dir.exists():
        logger.error(f"❌ Images directory not found: {images_dir}")
        return
    if not masks_dir.exists():
        logger.error(f"❌ Masks directory not found: {masks_dir}")
        return
    
    image_files = sorted(list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg')))
    mask_files = sorted(list(masks_dir.glob('*.png')) + list(masks_dir.glob('*.npy')))
    
    logger.info(f"✓ Images directory: {len(image_files)} files")
    logger.info(f"✓ Masks directory: {len(mask_files)} files")
    
    # Check alignment
    logger.info("\n2. CHECKING FILE ALIGNMENT")
    logger.info("-" * 80)
    
    mismatches = 0
    for img_file in image_files[:10]:
        mask_name = img_file.stem + '.npy'
        mask_file = masks_dir / mask_name
        if not mask_file.exists():
            logger.warning(f"  ⚠ Missing mask for: {img_file.name}")
            mismatches += 1
    
    if mismatches == 0:
        logger.info("✓ All sampled images have corresponding masks")
    else:
        logger.warning(f"⚠ Found {mismatches} mismatches in first 10 files")
    
    # Load and analyze data
    logger.info("\n3. LOADING DATA AND ANALYZING CLASSES")
    logger.info("-" * 80)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        images_dir=CONFIG['images_dir'],
        masks_dir=CONFIG['masks_dir'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        image_size=CONFIG['image_size'],
        num_classes=CONFIG['num_classes']
    )
    
    # Analyze class distribution
    logger.info("\n4. CLASS DISTRIBUTION ANALYSIS")
    logger.info("-" * 80)
    
    class_counts = Counter()
    image_shapes = Counter()
    mask_ranges = {'min': 255, 'max': 0}
    total_pixels = 0
    batch_count = 0
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 100:  # Sample first 100 batches
            break
        
        masks = batch['mask'].numpy()
        for mask in masks:
            # Record class distribution
            unique, counts = np.unique(mask, return_counts=True)
            for cls, count in zip(unique, counts):
                class_counts[int(cls)] += int(count)
                total_pixels += int(count)
            
            # Check mask ranges
            mask_ranges['min'] = min(mask_ranges['min'], int(mask.min()))
            mask_ranges['max'] = max(mask_ranges['max'], int(mask.max()))
            
            # Check image shape
            image_shapes[mask.shape] += 1
        
        batch_count += 1
    
    logger.info(f"\nAnalyzed {batch_count} batches:")
    logger.info(f"  Total pixels: {total_pixels:,}")
    logger.info(f"  Unique classes found: {len(class_counts)}")
    logger.info(f"  Mask value range: [{mask_ranges['min']}, {mask_ranges['max']}]")
    logger.info(f"  Image shapes: {dict(image_shapes)}")
    
    # Show class distribution
    logger.info("\n  Top 15 classes by pixel count:")
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    for cls, count in sorted_classes[:15]:
        pct = (count / total_pixels * 100) if total_pixels > 0 else 0
        logger.info(f"    Class {cls:3d}: {count:12,} pixels ({pct:6.2f}%)")
    
    logger.info("\n  Bottom 10 classes by pixel count:")
    for cls, count in sorted_classes[-10:]:
        pct = (count / total_pixels * 100) if total_pixels > 0 else 0
        logger.info(f"    Class {cls:3d}: {count:12,} pixels ({pct:6.2f}%)")
    
    # Check for problematic classes
    logger.info("\n5. CHECKING FOR PROBLEMATIC CLASSES")
    logger.info("-" * 80)
    
    zero_count_classes = [i for i in range(CONFIG['num_classes']) if i not in class_counts]
    logger.info(f"Classes with ZERO pixels: {len(zero_count_classes)}")
    if len(zero_count_classes) > 10:
        logger.warning(f"  Examples: {zero_count_classes[:10]}")
    
    # Check class 255 (often used as ignore/background)
    if 255 in class_counts:
        pct_255 = (class_counts[255] / total_pixels * 100)
        logger.warning(f"  Class 255 (often ignore): {class_counts[255]:,} pixels ({pct_255:.2f}%)")
    
    # Verify IoU calculation
    logger.info("\n6. VERIFYING IoU CALCULATION")
    logger.info("-" * 80)
    
    # Get a sample batch
    batch = next(iter(train_loader))
    masks = batch['mask'].numpy()[0]
    
    logger.info(f"Sample mask shape: {masks.shape}")
    logger.info(f"Sample mask unique values: {np.unique(masks)[:20]}")
    
    # Manually calculate IoU for Class 0
    logger.info("\n  Manual IoU calculation for Class 0:")
    class_0_pixels = (masks == 0).sum()
    logger.info(f"    Pixels of Class 0: {class_0_pixels:,}")
    
    # Create a dummy prediction (all zeros)
    pred_all_zero = np.zeros_like(masks)
    intersection = ((pred_all_zero == 0) & (masks == 0)).sum()
    union = ((pred_all_zero == 0) | (masks == 0)).sum()
    iou_all_zero = intersection / union if union > 0 else 0
    logger.info(f"    IoU if predict all 0: {iou_all_zero:.4f}")
    
    # Create a dummy prediction (all random)
    pred_random = np.random.randint(0, 256, masks.shape)
    intersection = ((pred_random == 0) & (masks == 0)).sum()
    union = ((pred_random == 0) | (masks == 0)).sum()
    iou_random = intersection / union if union > 0 else 0
    logger.info(f"    IoU if predict random: {iou_random:.4f}")
    
    # Visualize a sample
    logger.info("\n7. VISUALIZING SAMPLE DATA")
    logger.info("-" * 80)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    for idx, batch in enumerate(train_loader):
        if idx >= 3:
            break
        
        image = batch['image'][0].numpy()
        mask = batch['mask'][0].numpy()
        
        # Denormalize image
        image = np.clip(image.transpose(1, 2, 0) * 0.5 + 0.5, 0, 1)
        
        # Plot image
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title(f'Image {idx}')
        axes[idx, 0].axis('off')
        
        # Plot mask
        axes[idx, 1].imshow(mask, cmap='tab20')
        axes[idx, 1].set_title(f'Mask {idx} (unique: {len(np.unique(mask))})')
        axes[idx, 1].axis('off')
        
        # Plot mask with color intensity
        axes[idx, 2].imshow(mask, cmap='gray')
        axes[idx, 2].set_title(f'Mask {idx} (gray)')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    viz_path = Path('models/checkpoints_fixed/data_quality_viz.png')
    plt.savefig(viz_path, dpi=100, bbox_inches='tight')
    logger.info(f"✓ Visualization saved to: {viz_path}")
    plt.close()
    
    # Generate report
    logger.info("\n8. GENERATING SUMMARY REPORT")
    logger.info("-" * 80)
    
    report = {
        'total_images': len(image_files),
        'total_masks': len(mask_files),
        'batches_analyzed': batch_count,
        'total_pixels': total_pixels,
        'unique_classes': len(class_counts),
        'classes_with_zero_pixels': len(zero_count_classes),
        'mask_value_range': [int(mask_ranges['min']), int(mask_ranges['max'])],
        'top_5_classes': [(int(c), int(cnt), float(cnt/total_pixels*100)) for c, cnt in sorted_classes[:5]],
        'class_255_pixels': int(class_counts.get(255, 0)),
        'class_255_percentage': float(class_counts.get(255, 0) / total_pixels * 100) if total_pixels > 0 else 0.0
    }
    
    report_path = Path('models/checkpoints_fixed/data_quality_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("\n📊 DIAGNOSTIC SUMMARY:")
    logger.info(f"  • Total images: {report['total_images']}")
    logger.info(f"  • Unique classes: {report['unique_classes']} / 256")
    logger.info(f"  • Classes with pixels: {report['unique_classes']}")
    logger.info(f"  • Classes with ZERO pixels: {report['classes_with_zero_pixels']}")
    logger.info(f"  • Mask value range: {report['mask_value_range']}")
    logger.info(f"  • Class 255 (ignore?): {report['class_255_percentage']:.2f}%")
    
    logger.info(f"\n✓ Full report saved to: {report_path}")
    
    logger.info("\n" + "="*80)
    logger.info("DIAGNOSIS COMPLETE")
    logger.info("="*80)
    
    # Interpretation
    logger.info("\n📋 INTERPRETATION GUIDE:")
    logger.info("-" * 80)
    
    if report['unique_classes'] < 10:
        logger.error("❌ ISSUE DETECTED: Very few classes have pixels!")
        logger.error("   This means most classes are not present in the data.")
        logger.error("   ACTION: Check if annotations are correct or if you need different data.")
    elif report['class_255_percentage'] > 50:
        logger.error("❌ ISSUE DETECTED: Class 255 dominates the data!")
        logger.error("   This class usually means 'ignore' or 'background'.")
        logger.error("   ACTION: Check if masks need remapping (255 → 0).")
    elif report['classes_with_zero_pixels'] > 200:
        logger.warning("⚠️ WARNING: Many classes have zero pixels.")
        logger.warning("   This is expected for 256-class problem, but check if intentional.")
    else:
        logger.info("✓ Data looks reasonable for training.")
        logger.info("   If training still fails, the issue is likely in:")
        logger.info("   - Weight calculation approach")
        logger.info("   - Model architecture")
        logger.info("   - Learning rate / optimization")
    
    return report


if __name__ == "__main__":
    try:
        report = diagnose_data_quality()
    except Exception as e:
        logger.error(f"\n\nDiagnostics failed: {e}", exc_info=True)
