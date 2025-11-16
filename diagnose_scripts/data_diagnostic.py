#!/usr/bin/env python3
"""
Comprehensive Data Diagnostic Script
Identifies issues with ground truth masks, class distribution, and data quality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import json
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import os

def check_dataset_structure(images_dir, masks_dir):
    """Check if dataset structure is valid"""
    print("\n" + "="*80)
    print("STEP 1: DATASET STRUCTURE VALIDATION")
    print("="*80)
    
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return False
    if not masks_dir.exists():
        print(f"❌ Masks directory not found: {masks_dir}")
        return False
    
    image_files = sorted(list(images_dir.glob("*.png")))
    mask_files = sorted(list(masks_dir.glob("*.png")))
    
    print(f"✓ Images found: {len(image_files)}")
    print(f"✓ Masks found: {len(mask_files)}")
    
    if len(image_files) != len(mask_files):
        print(f"⚠️  MISMATCH: {len(image_files)} images vs {len(mask_files)} masks")
        return False
    
    if len(image_files) == 0:
        print("❌ No files found in dataset!")
        return False
    
    print(f"✓ Dataset structure valid")
    return True

def validate_images_and_masks(images_dir, masks_dir, num_samples=10):
    """Validate image/mask alignment and dimensions"""
    print("\n" + "="*80)
    print(f"STEP 2: IMAGE/MASK VALIDATION ({num_samples} samples)")
    print("="*80)
    
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    
    image_files = sorted(list(images_dir.glob("*.png")))[:num_samples]
    
    issues = []
    
    for img_path in image_files:
        mask_path = masks_dir / img_path.name
        
        if not mask_path.exists():
            issues.append(f"Missing mask: {img_path.name}")
            continue
        
        try:
            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                issues.append(f"Corrupted image: {img_path.name}")
                continue
            if mask is None:
                issues.append(f"Corrupted mask: {img_path.name}")
                continue
            
            # Check dimensions
            if img.shape[:2] != mask.shape[:2]:
                issues.append(f"Dimension mismatch {img_path.name}: {img.shape} vs {mask.shape}")
            
            # Check if mask is all zeros
            if mask.max() == 0:
                issues.append(f"Empty mask (all zeros): {img_path.name}")
            
            # Check if image is all zeros
            if img.max() == 0:
                issues.append(f"Empty image (all zeros): {img_path.name}")
                
        except Exception as e:
            issues.append(f"Error reading {img_path.name}: {str(e)}")
    
    if issues:
        print(f"⚠️  Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"   - {issue}")
        if len(issues) > 10:
            print(f"   ... and {len(issues) - 10} more")
    else:
        print(f"✓ All {num_samples} samples valid")
    
    return len(issues) == 0

def analyze_class_distribution(masks_dir, num_classes=12):
    """Analyze class distribution in masks"""
    print("\n" + "="*80)
    print("STEP 3: CLASS DISTRIBUTION ANALYSIS")
    print("="*80)
    
    masks_dir = Path(masks_dir)
    mask_files = sorted(list(masks_dir.glob("*.png")))
    
    class_counts = Counter()
    total_pixels = 0
    empty_masks = 0
    
    print(f"Analyzing {len(mask_files)} masks...")
    
    for mask_path in mask_files:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            continue
        
        if mask.max() == 0:
            empty_masks += 1
            continue
        
        unique, counts = np.unique(mask, return_counts=True)
        for cls, count in zip(unique, counts):
            class_counts[int(cls)] += int(count)
            total_pixels += int(count)
    
    print(f"✓ Total pixels analyzed: {total_pixels:,}")
    print(f"⚠️  Empty masks found: {empty_masks}/{len(mask_files)}")
    
    if total_pixels == 0:
        print("❌ CRITICAL: No valid pixel data in masks!")
        return False
    
    print(f"\nClass Distribution (pixels):")
    print(f"{'Class':<8} {'Pixels':<15} {'%':<10}")
    print("-" * 35)
    
    for cls in range(num_classes):
        count = class_counts.get(cls, 0)
        pct = (count / total_pixels * 100) if total_pixels > 0 else 0
        status = "✓" if count > 0 else "✗"
        print(f"{cls:<8} {count:<15,} {pct:<10.2f}% {status}")
    
    # Calculate imbalance ratio
    class_counts_list = [class_counts.get(i, 0) for i in range(num_classes)]
    nonzero_counts = [c for c in class_counts_list if c > 0]
    if nonzero_counts:
        imbalance_ratio = max(nonzero_counts) / min(nonzero_counts)
        print(f"\nClass Imbalance Ratio: {imbalance_ratio:.1f}:1")
    
    return True

def visualize_samples(images_dir, masks_dir, num_samples=5):
    """Visualize random samples with images and masks side-by-side"""
    print("\n" + "="*80)
    print(f"STEP 4: VISUALIZING {num_samples} SAMPLES")
    print("="*80)
    
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    
    image_files = sorted(list(images_dir.glob("*.png")))
    
    # Select random samples
    import random
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_path in enumerate(sample_files):
        mask_path = masks_dir / img_path.name
        
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Display image
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f"Image: {img_path.name}")
        axes[idx, 0].axis('off')
        
        # Display mask with class colors
        mask_colored = plt.cm.tab20(mask / 20)
        axes[idx, 1].imshow(mask_colored)
        axes[idx, 1].set_title(f"Mask: Classes {sorted(np.unique(mask))}")
        axes[idx, 1].axis('off')
    
    plt.tight_layout()
    output_path = Path('VpC3/outputs/data_diagnostic_samples.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")
    plt.close()

def check_preprocessing(images_dir, masks_dir):
    """Check preprocessing issues"""
    print("\n" + "="*80)
    print("STEP 5: PREPROCESSING VALIDATION")
    print("="*80)
    
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    
    # Sample one image and mask
    image_files = list(images_dir.glob("*.png"))
    if not image_files:
        print("❌ No images found")
        return False
    
    img_path = image_files[0]
    mask_path = masks_dir / img_path.name
    
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    print(f"Sample: {img_path.name}")
    print(f"  Image shape: {img.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Image dtype: {img.dtype}")
    print(f"  Mask dtype: {mask.dtype}")
    print(f"  Image range: [{img.min()}, {img.max()}]")
    print(f"  Mask range: [{mask.min()}, {mask.max()}]")
    
    # Check if dimensions are 512x512
    if img.shape[0] != 512 or img.shape[1] != 512:
        print(f"⚠️  Expected 512x512, got {img.shape[0]}x{img.shape[1]}")
    
    # Check normalization
    if img.max() <= 1:
        print("⚠️  Images may be normalized to [0,1] instead of [0,255]")
    
    print("✓ Preprocessing check complete")
    return True

def generate_report(images_dir, masks_dir):
    """Generate comprehensive diagnostic report"""
    print("\n" + "="*80)
    print("DATA DIAGNOSTIC REPORT")
    print("="*80)
    
    # Run all diagnostics
    struct_ok = check_dataset_structure(images_dir, masks_dir)
    valid_ok = validate_images_and_masks(images_dir, masks_dir, num_samples=20)
    class_ok = analyze_class_distribution(masks_dir, num_classes=12)
    visualize_samples(images_dir, masks_dir, num_samples=5)
    preproc_ok = check_preprocessing(images_dir, masks_dir)
    
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    print(f"Structure Valid:     {'✓' if struct_ok else '❌'}")
    print(f"Images/Masks Valid:  {'✓' if valid_ok else '❌'}")
    print(f"Classes Present:     {'✓' if class_ok else '❌'}")
    print(f"Preprocessing OK:    {'✓' if preproc_ok else '❌'}")
    
    all_ok = struct_ok and valid_ok and class_ok and preproc_ok
    
    print("\n" + "="*80)
    if all_ok:
        print("✅ DATA QUALITY: GOOD - Proceed with model debugging")
    else:
        print("❌ DATA QUALITY: ISSUES FOUND - Fix data before training")
    print("="*80)
    
    return all_ok

if __name__ == "__main__":
    images_dir = 'data/processed/images'
    masks_dir = 'data/processed/annotations'
    
    print("\n" + "="*80)
    print("FLOOR PLAN DATASET DIAGNOSTIC TOOL")
    print("="*80)
    print(f"Images: {images_dir}")
    print(f"Masks:  {masks_dir}")
    
    result = generate_report(images_dir, masks_dir)
    
    if result:
        print("\n✅ Data appears to be correctly preprocessed")
        print("   Problem is likely in model training/architecture")
    else:
        print("\n❌ Data has issues that need to be fixed")
        print("   Regenerate processed dataset or check preprocessing script")
