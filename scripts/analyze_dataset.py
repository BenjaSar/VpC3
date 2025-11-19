#!/usr/bin/env python3
"""
Dataset Analysis Script
Analyzes class distribution and imbalance in training data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
from tqdm import tqdm
from collections import Counter
import json

def analyze_class_distribution(masks_dir, num_classes=12):
    """
    Analyze class distribution across entire dataset
    
    Args:
        masks_dir: Path to masks directory
        num_classes: Number of classes
    
    Returns:
        Dictionary with class statistics
    """
    masks_path = Path(masks_dir)
    mask_files = sorted([f for f in masks_path.glob('*.png')])
    
    print(f"Analyzing {len(mask_files)} mask files...\n")
    
    class_counts = Counter()
    total_pixels = 0
    per_image_stats = []
    
    for mask_file in tqdm(mask_files, desc="Processing masks"):
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        unique, counts = np.unique(mask, return_counts=True)
        for cls, count in zip(unique, counts):
            class_counts[int(cls)] += int(count)
            total_pixels += int(count)
        
        per_image_stats.append({
            'file': mask_file.name,
            'total_pixels': len(mask.flatten()),
            'unique_classes': len(unique)
        })
    
    # Calculate percentages and weights
    print("\n" + "="*80)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*80 + "\n")
    
    results = {
        'total_images': len(mask_files),
        'total_pixels': total_pixels,
        'class_statistics': {}
    }
    
    for cls in range(num_classes):
        count = class_counts.get(cls, 0)
        percentage = (count / total_pixels * 100) if total_pixels > 0 else 0
        weight = (total_pixels / (num_classes * count)) if count > 0 else 0
        weight = min(weight, 100.0)  # Cap weight
        
        results['class_statistics'][f'class_{cls}'] = {
            'pixel_count': int(count),
            'percentage': round(percentage, 4),
            'weight': round(weight, 4)
        }
        
        print(f"Class {cls:2d}: {count:10,d} pixels ({percentage:6.2f}%) | Weight: {weight:7.2f}")
    
    # Calculate imbalance ratios
    print("\n" + "-"*80)
    print("IMBALANCE RATIOS (vs background/class 0):")
    print("-"*80 + "\n")
    
    class_0_count = class_counts.get(0, 1)
    ratios = {}
    
    for cls in range(1, num_classes):
        count = class_counts.get(cls, 1)
        ratio = class_0_count / count if count > 0 else float('inf')
        ratios[f'class_0_vs_class_{cls}'] = round(ratio, 1)
        print(f"Class 0 vs Class {cls:2d}: {ratio:8.1f}:1")
    
    results['imbalance_ratios'] = ratios
    
    # Find worst imbalance
    max_ratio = max(ratios.values()) if ratios else 0
    print(f"\nMax Imbalance Ratio: {max_ratio}:1")
    
    # Statistics
    print("\n" + "-"*80)
    print("DATASET STATISTICS:")
    print("-"*80 + "\n")
    
    print(f"Total Images: {len(mask_files)}")
    print(f"Total Pixels: {total_pixels:,}")
    print(f"Avg Pixels/Image: {total_pixels/len(mask_files):,.0f}")
    print(f"Background (Class 0) Coverage: {(class_counts[0]/total_pixels*100):.2f}%")
    print(f"Active Classes (>0 pixels): {sum(1 for c in class_counts.values() if c > 0)}/{num_classes}")
    
    results['summary'] = {
        'total_images': len(mask_files),
        'total_pixels': int(total_pixels),
        'avg_pixels_per_image': int(total_pixels / len(mask_files)) if len(mask_files) > 0 else 0,
        'background_coverage_percent': round(class_counts[0] / total_pixels * 100, 2) if total_pixels > 0 else 0,
        'active_classes': sum(1 for c in class_counts.values() if c > 0),
        'max_imbalance_ratio': round(max_ratio, 1)
    }
    
    return results

def main():
    masks_dir = 'data/processed/annotations'
    
    if not Path(masks_dir).exists():
        print(f"Error: Masks directory not found: {masks_dir}")
        return
    
    results = analyze_class_distribution(masks_dir)
    
    # Save results
    output_file = 'VpC3/outputs/dataset_analysis.json'
    Path('VpC3/outputs').mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Analysis saved to: {output_file}")

if __name__ == "__main__":
    main()
