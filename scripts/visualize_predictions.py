#!/usr/bin/env python3
"""
Diagnostic Script: Visualization of Model Predictions
Shows what the model predicts vs ground truth masks
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import cv2

from src.data.dataset import create_dataloaders
from src.utils.logging_config import setup_logging
from models.vit_segmentation import ViTSegmentation

logger = setup_logging()

# Define colors for 12 classes
COLORS = [
    (0, 0, 0),           # 0: Background - Black
    (255, 0, 0),         # 1: Walls - Red
    (0, 255, 0),         # 2: Kitchen - Green
    (0, 0, 255),         # 3: Living Room - Blue
    (255, 255, 0),       # 4: Bedroom - Yellow
    (255, 0, 255),       # 5: Bathroom - Magenta
    (0, 255, 255),       # 6: Hallway - Cyan
    (128, 0, 0),         # 7: Storage - Dark Red
    (0, 128, 0),         # 8: Garage - Dark Green
    (0, 0, 128),         # 9: Undefined - Dark Blue
    (128, 128, 0),       # 10: Closet - Dark Yellow
    (128, 0, 128),       # 11: Balcony - Dark Magenta
]

CLASS_NAMES = [
    "Background", "Walls", "Kitchen", "Living Room",
    "Bedroom", "Bathroom", "Hallway", "Storage",
    "Garage", "Undefined", "Closet", "Balcony"
]

def visualize_predictions():
    """Visualize model predictions on sample batch"""
    
    logger.info("=" * 80)
    logger.info("PREDICTION VISUALIZATION")
    logger.info("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataloader
    logger.info("Creating dataloader...")
    try:
        train_loader, _, _ = create_dataloaders(
            images_dir='data/processed/images',
            masks_dir='data/processed/annotations',
            batch_size=4,
            num_workers=0,
            image_size=512,
            num_classes=12
        )
    except Exception as e:
        logger.error(f"Failed to create dataloader: {e}")
        return
    
    # Load a pretrained model if available
    checkpoint_path = Path("models/checkpoints/best_model.pth")
    
    if checkpoint_path.exists():
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        model = ViTSegmentation(n_classes=12).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model loaded from checkpoint")
    else:
        logger.info("No checkpoint found, using randomly initialized model")
        model = ViTSegmentation(n_classes=12).to(device)
    
    model.eval()
    
    # Get a batch
    batch = next(iter(train_loader))
    images = batch['image'].to(device)
    masks_true = batch['mask'].cpu().numpy()
    filenames = batch['filename']
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        masks_pred = outputs.argmax(dim=1).cpu().numpy()
    
    logger.info(f"\nVisualizing {len(images)} samples...")
    
    # Create figure
    num_samples = min(4, len(images))
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Original image
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # Ground truth mask
        mask_true = masks_true[i]
        
        # Predicted mask
        mask_pred = masks_pred[i]
        
        # Convert to colored images
        mask_true_colored = np.zeros((*mask_true.shape, 3), dtype=np.uint8)
        mask_pred_colored = np.zeros((*mask_pred.shape, 3), dtype=np.uint8)
        
        for cls in range(12):
            mask_true_colored[mask_true == cls] = COLORS[cls]
            mask_pred_colored[mask_pred == cls] = COLORS[cls]
        
        # Plot
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Image: {filenames[i]}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask_true_colored)
        axes[i, 1].set_title(f"Ground Truth")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(mask_pred_colored)
        axes[i, 2].set_title(f"Prediction")
        axes[i, 2].axis('off')
        
        # Calculate accuracy
        acc = np.mean(mask_true == mask_pred)
        logger.info(f"Sample {i}: Pixel accuracy = {acc:.4f}")
        
        # Class distribution
        logger.info(f"  Ground truth classes: {np.unique(mask_true).tolist()}")
        logger.info(f"  Predicted classes: {np.unique(mask_pred).tolist()}")
    
    # Add legend
    legend_elements = [mpatches.Patch(facecolor=np.array(COLORS[i])/255, 
                                      label=CLASS_NAMES[i]) 
                       for i in range(12)]
    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, -0.02), ncol=6, fontsize=8)
    
    plt.tight_layout()
    output_path = Path("prediction_visualization.png")
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    logger.info(f"\n✓ Visualization saved to: {output_path}")
    
    # Additional analysis
    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION ANALYSIS")
    logger.info("=" * 80)
    
    all_pred = masks_pred.flatten()
    class_counts = np.bincount(all_pred, minlength=12)
    
    logger.info("\nPredicted class distribution:")
    for cls in range(12):
        count = class_counts[cls]
        pct = (count / len(all_pred) * 100)
        bar = "█" * int(pct / 5)
        logger.info(f"  Class {cls:2d} ({CLASS_NAMES[cls]:15s}): {count:8d} ({pct:5.2f}%) {bar}")
    
    # Check if model is predicting only one class
    unique_pred = np.unique(all_pred)
    if len(unique_pred) == 1:
        logger.warning(f"\n⚠️  Model is predicting ONLY class {unique_pred[0]}!")
        logger.warning("This is a critical issue - the model is not learning!")
    elif len(unique_pred) < 3:
        logger.warning(f"\n⚠️  Model is predicting very few classes: {unique_pred.tolist()}")

if __name__ == "__main__":
    try:
        visualize_predictions()
    except Exception as e:
        logger.error(f"Visualization failed: {e}", exc_info=True)
