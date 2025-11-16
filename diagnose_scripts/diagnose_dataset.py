#!/usr/bin/env python3
"""
Diagnostic Script: Dataset Integrity Check
Verifies image-mask correspondence and class distributions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt

def diagnose_dataset():
    """Comprehensive dataset diagnostics"""
    
    images_dir = Path("data/processed/images")
    masks_dir = Path("data/processed/annotations")
    
    if not images_dir.exists() or not masks_dir.exists():
        print("❌ Processed dataset not found!")
        print(f"   Images: {images_dir.exists()}")
        print(f"   Masks: {masks_dir.exists()}")
        return
    
    print("=" * 80)
    print("DATASET DIAGNOSTICS")
    print("=" * 80)
    
    # Get all files
    image_files = sorted(list(images_dir.glob("*.png")))
    mask_files = sorted(list(masks_dir.glob("*.png")))
    
    print(f"\n✓ Found {len(image_files)} images")
    print(f"✓ Found {len(mask_files)} masks")
    
    # Check correspondence
    print("\n--- IMAGE-MASK CORRESPONDENCE ---")
    missing_masks = []
    for img_file in image_files[:5]:  # Check first 5
        mask_file = masks_dir / img_file.name
        if mask_file.exists():
            print(f"✓ {img_file.name}")
        else:
            print(f"❌ {img_file.name} - NO MASK")
            missing_masks.append(img_file.name)
    
    if missing_masks:
        print(f"\n⚠️  Found {len(missing_masks)} images with missing masks!")
    
    # Check shapes
    print("\n--- IMAGE SHAPES ---")
    shapes = []
    for img_file in image_files[:10]:
        img = cv2.imread(str(img_file))
        mask = cv2.imread(str(masks_dir / img_file.name), cv2.IMREAD_GRAYSCALE)
        if img is not None and mask is not None:
            shapes.append((img.shape, mask.shape))
            if len(shapes) <= 3:
                print(f"  {img_file.name}: image {img.shape}, mask {mask.shape}")
    
    # Check for shape consistency
    if shapes:
        consistent = all(s[0][:2] == s[1][:2] for s in shapes)
        if consistent:
            print(f"✓ All shapes consistent")
        else:
            print(f"❌ Shape mismatch detected!")
    
    # Check class values
    print("\n--- CLASS VALUE RANGES ---")
    all_classes = Counter()
    min_val, max_val = 255, 0
    
    for mask_file in mask_files[:50]:  # Sample 50 masks
        mask = np.array(Image.open(mask_file))
        all_classes.update(mask.flatten())
        min_val = min(min_val, mask.min())
        max_val = max(max_val, mask.max())
    
    print(f"  Min value: {min_val}")
    print(f"  Max value: {max_val}")
    print(f"  Expected range: [0, 11]")
    
    if max_val > 11 or min_val < 0:
        print(f"❌ VALUES OUT OF RANGE!")
    else:
        print(f"✓ All values in valid range")
    
    # Class distribution
    print(f"\n--- CLASS DISTRIBUTION (sample of 50 masks) ---")
    total_pixels = sum(all_classes.values())
    for cls in sorted(all_classes.keys()):
        count = all_classes[cls]
        pct = (count / total_pixels * 100)
        bar = "█" * int(pct / 2)
        print(f"  Class {cls:2d}: {count:8d} pixels ({pct:5.2f}%) {bar}")
    
    # Check for dominant class
    if all_classes:
        most_common_cls, most_common_count = all_classes.most_common(1)[0]
        dominant_pct = (most_common_count / total_pixels * 100)
        print(f"\n  Dominant class: {most_common_cls} ({dominant_pct:.2f}%)")
        if dominant_pct > 50:
            print(f"  ⚠️  Very high dominant class ratio!")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    issues = []
    if missing_masks:
        issues.append(f"❌ {len(missing_masks)} images missing masks")
    if max_val > 11 or min_val < 0:
        issues.append(f"❌ Class values out of range [{min_val}, {max_val}]")
    if not shapes or not all(s[0][:2] == s[1][:2] for s in shapes):
        issues.append(f"❌ Image-mask shape mismatch")
    
    if not issues:
        print("✓ Dataset appears to be correct!")
    else:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")

if __name__ == "__main__":
    diagnose_dataset()
