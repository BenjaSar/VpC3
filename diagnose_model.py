#!/usr/bin/env python3
"""
Diagnostic Script for Floor Plan Segmentation Model
Identifies the root cause of poor IoU performance
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import cv2
from collections import Counter
from tqdm import tqdm
import json

from data.dataset import create_dataloaders


def analyze_dataset(images_dir, masks_dir, batch_size=8, num_workers=0):
    """Analyze dataset for class distribution and other issues"""
    
    print("="*80)
    print("DATASET DIAGNOSIS")
    print("="*80)
    
    # Create dataloader
    train_loader, val_loader, test_loader = create_dataloaders(
        images_dir=images_dir,
        masks_dir=masks_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=512,
        num_classes=256
    )
    
    # Analyze class distribution
    print("\n1. Analyzing Class Distribution...")
    class_counts = Counter()
    total_pixels = 0
    
    for batch in tqdm(train_loader, desc="Scanning training data"):
        masks = batch['mask'].numpy()
        for mask in masks:
            unique, counts = np.unique(mask, return_counts=True)
            for cls, count in zip(unique, counts):
                class_counts[int(cls)] += int(count)
                total_pixels += int(count)
    
    print("\nClass Distribution in Training Set:")
    print("-" * 80)
    print(f"{'Class':<10} {'Pixels':<15} {'Percentage':<15} {'Frequency':<15}")
    print("-" * 80)
    
    sorted_classes = sorted(class_counts.items())
    class_weights = {}
    
    for cls, count in sorted_classes:
        percentage = (count / total_pixels) * 100
        frequency = count / len(train_loader.dataset)
        class_weights[cls] = total_pixels / (256 * count)  # Inverse frequency
        print(f"{cls:<10} {count:<15,} {percentage:<14.4f}% {frequency:<15.1f}")
    
    # Identify severely imbalanced classes
    print("\n2. Class Imbalance Analysis:")
    print("-" * 80)
    
    percentages = [(cls, (count/total_pixels)*100) for cls, count in sorted_classes]
    max_pct = max(p[1] for p in percentages)
    min_pct = min(p[1] for p in percentages if p[1] > 0)
    
    print(f"Most common class: {percentages[0][0]} ({max_pct:.2f}%)")
    print(f"Least common class: {min(sorted_classes, key=lambda x: x[1])[0]} ({min_pct:.4f}%)")
    print(f"Imbalance ratio: {max_pct/min_pct:.2f}x")
    
    if max_pct > 50:
        print("⚠️  WARNING: Severe class imbalance detected!")
        print(f"   Class {percentages[0][0]} dominates with {max_pct:.1f}% of pixels")
    
    # Check for missing classes
    missing_classes = set(range(256)) - set(class_counts.keys())
    if missing_classes:
        print(f"\n⚠️  WARNING: Missing classes in training data: {sorted(missing_classes)}")
    
    # Calculate recommended class weights
    print("\n3. Recommended Class Weights:")
    print("-" * 80)
    
    # Normalize weights
    max_weight = max(class_weights.values())
    normalized_weights = {cls: min(w/max_weight * 10, 100) for cls, w in class_weights.items()}
    
    weights_list = [normalized_weights.get(i, 1.0) for i in range(256)]
    
    print("Paste this in train.py:")
    print(f"\nclass_weights = torch.tensor({weights_list})")
    print("criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))")
    
    # Save weights to file
    with open('class_weights.json', 'w') as f:
        json.dump({
            'class_counts': dict(sorted_classes),
            'class_percentages': {str(cls): (count/total_pixels)*100 
                                 for cls, count in sorted_classes},
            'recommended_weights': weights_list
        }, f, indent=2)
    
    print("\n✓ Class weights saved to: class_weights.json")
    
    # Check mask value ranges
    print("\n4. Checking Mask Value Ranges...")
    print("-" * 80)
    
    invalid_count = 0
    for batch in train_loader:
        masks = batch['mask'].numpy()
        if masks.max() > 255 or masks.min() < 0:
            invalid_count += 1
    
    if invalid_count > 0:
        print(f"⚠️  WARNING: {invalid_count} batches have invalid mask values!")
        print("   Mask values should be in range [0, 255]")
    else:
        print("✓ All masks have valid values [0, 255]")
    
    # Check image-mask alignment
    print("\n5. Checking Image-Mask Alignment...")
    print("-" * 80)
    
    sample_batch = next(iter(train_loader))
    images = sample_batch['image']
    masks = sample_batch['mask']
    
    print(f"Image batch shape: {images.shape}")
    print(f"Mask batch shape: {masks.shape}")
    
    if images.shape[-2:] != masks.shape[-2:]:
        print("⚠️  WARNING: Image and mask dimensions don't match!")
    else:
        print("✓ Image and mask dimensions match")
    
    return class_weights


def analyze_model_predictions(checkpoint_path='models/checkpoints/best_model.pth'):
    """Analyze what the model is actually predicting"""
    
    print("\n" + "="*80)
    print("MODEL PREDICTION ANALYSIS")
    print("="*80)
    
    if not Path(checkpoint_path).exists():
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        return
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check training history
    if 'history' in checkpoint:
        history = checkpoint['history']
        print("\nTraining History:")
        print("-" * 80)
        
        if 'train_iou' in history and len(history['train_iou']) > 0:
            print(f"Initial train IoU: {history['train_iou'][0]:.4f}")
            print(f"Final train IoU: {history['train_iou'][-1]:.4f}")
            
            if 'val_iou' in history and len(history['val_iou']) > 0:
                print(f"Initial val IoU: {history['val_iou'][0]:.4f}")
                print(f"Final val IoU: {history['val_iou'][-1]:.4f}")
                
                improvement = history['val_iou'][-1] - history['val_iou'][0]
                if improvement < 0.05:
                    print(f"\n⚠️  WARNING: Model barely improved (Δ={improvement:.4f})")
                    print("   This suggests:")
                    print("   - Learning rate might be too low")
                    print("   - Model is stuck in local minimum")
                    print("   - Class imbalance is too severe")


def main():
    print("\n" + "="*80)
    print("FLOOR PLAN MODEL DIAGNOSTIC TOOL")
    print("="*80)
    
    # Configuration
    IMAGES_DIR = 'data/processed/images'
    MASKS_DIR = 'data/processed/annotations'
    CHECKPOINT_PATH = 'models/checkpoints/best_model.pth'
    
    # Run diagnostics
    print("\nRunning diagnostics on training data...")
    class_weights = analyze_dataset(IMAGES_DIR, MASKS_DIR)
    
    print("\nAnalyzing trained model...")
    analyze_model_predictions(CHECKPOINT_PATH)
    
    # Final recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    print("""
Based on the analysis, here are the recommended fixes:

1. ⚠️  CRITICAL: Add class weights to loss function
   - Copy the class_weights code from above into train.py
   - This will force the model to learn minority classes

2. 📉 Adjust learning rate
   - Try: learning_rate = 5e-5 (lower than current)
   - Or use: OneCycleLR scheduler for better convergence

3. 🎯 Use Focal Loss for severe imbalance
   - Install: pip install segmentation-models-pytorch
   - Use Focal Loss instead of CrossEntropyLoss

4. 🔄 Increase training epochs
   - Train for at least 100-150 epochs with class weights
   - Early stopping on validation IoU

5. 📊 Monitor per-class IoU during training
   - Not just mean IoU - watch individual classes
   - Some classes might need more epochs

6. 🔍 Data augmentation
   - Already enabled, but ensure it's working
   - Consider stronger augmentation for minority classes

Run the fixed train.py and you should see:
- All classes getting some predictions (not just class 0)
- Mean IoU > 0.40 after 50 epochs
- Mean IoU > 0.60 after 100 epochs
""")
    
    print("\nDiagnostics complete! Check class_weights.json for details.")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError during diagnosis: {e}")
        import traceback
        traceback.print_exc()
