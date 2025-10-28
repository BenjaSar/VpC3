#!/usr/bin/env python3
"""
Inference Script for ViT-Small Floor Plan Segmentation
Test trained model on test set or individual images
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from train import ViTSegmentation
from data.dataset import create_dataloaders
from src.utils.logging_config import setup_logging

logger = setup_logging()


# CubiCasa5K color map for visualization (256 classes)
COLOR_MAP = {
    0: [0, 0, 0],        # Background
    1: [192, 192, 192],  # Outdoor
    2: [128, 0, 0],      # Wall
    3: [128, 64, 128],   # Kitchen
    4: [0, 128, 0],      # Living room
    5: [128, 128, 0],    # Bedroom
    6: [0, 0, 128],      # Bath
    7: [128, 0, 128],    # Entry
    8: [0, 128, 128],    # Railing
    9: [64, 0, 0],       # Storage
    10: [192, 0, 0],     # Garage
    11: [64, 128, 0],    # Undefined
    12: [192, 128, 0],   # Interior door
    13: [64, 0, 128],    # Exterior door
    14: [192, 0, 128],   # Window
    15: [64, 128, 128],  # Other rooms
    255: [255, 255, 255],  # Background/White
}

# Fill remaining classes with random colors
np.random.seed(42)  # For consistency
for i in range(16, 256):
    if i not in COLOR_MAP:
        COLOR_MAP[i] = list(np.random.randint(0, 255, 3))


def load_model(checkpoint_path: str, device: torch.device):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = ViTSegmentation(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        in_channels=3,
        n_classes=config['n_classes'],
        embed_dim=config['embed_dim'],
        n_encoder_layers=config['n_encoder_layers'],
        n_decoder_layers=config['n_decoder_layers'],
        n_heads=config['n_heads'],
        mlp_ratio=config['mlp_ratio'],
        dropout=config['dropout']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Model loaded successfully")
    logger.info(f"Trained for {checkpoint['epoch'] + 1} epochs")
    if 'val_iou' in checkpoint:
        logger.info(f"Validation IoU: {checkpoint['val_iou']:.4f}")
    
    return model, config


def colorize_mask(mask: np.ndarray, color_map: dict) -> np.ndarray:
    """
    Convert class mask to RGB image
    
    Args:
        mask: Class mask (H, W)
        color_map: Dictionary mapping class IDs to RGB colors
        
    Returns:
        RGB image (H, W, 3)
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in color_map.items():
        color_mask[mask == class_id] = color
    
    return color_mask


def calculate_metrics(pred: np.ndarray, target: np.ndarray, n_classes: int):
    """
    Calculate segmentation metrics
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        n_classes: Number of classes
        
    Returns:
        Dictionary of metrics
    """
    pred = pred.flatten()
    target = target.flatten()
    
    # Overall accuracy
    accuracy = (pred == target).sum() / len(pred)
    
    # Per-class IoU
    ious = []
    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    
    mean_iou = np.nanmean(ious)
    
    return {
        'accuracy': accuracy,
        'mean_iou': mean_iou,
        'per_class_iou': ious
    }


def predict_single_image(model, image_path: str, device: torch.device, 
                        img_size: int = 512):
    """
    Predict on a single image
    
    Args:
        model: Trained model
        image_path: Path to input image
        device: Device
        img_size: Image size
        
    Returns:
        Predicted mask
    """
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    
    # Resize
    image = cv2.resize(image, (img_size, img_size))
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # To tensor
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    image = image.to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1).squeeze(0).cpu().numpy()
    
    # Resize to original size
    pred = cv2.resize(pred.astype(np.uint8), 
                     (original_size[1], original_size[0]), 
                     interpolation=cv2.INTER_NEAREST)
    
    return pred


def visualize_prediction(image_path: str, pred_mask: np.ndarray, 
                        gt_mask_path: str = None, output_path: str = None):
    """
    Visualize prediction results
    
    Args:
        image_path: Path to input image
        pred_mask: Predicted mask
        gt_mask_path: Path to ground truth mask (optional)
        output_path: Path to save visualization (optional)
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Colorize prediction
    pred_colored = colorize_mask(pred_mask, COLOR_MAP)
    
    # Create figure
    if gt_mask_path and Path(gt_mask_path).exists():
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Ground truth
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        gt_colored = colorize_mask(gt_mask, COLOR_MAP)
        
        axes[0].imshow(image)
        axes[0].set_title('Input Image', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(gt_colored)
        axes[1].set_title('Ground Truth', fontsize=14)
        axes[1].axis('off')
        
        axes[2].imshow(pred_colored)
        axes[2].set_title('Prediction', fontsize=14)
        axes[2].axis('off')
        
        # Calculate metrics
        metrics = calculate_metrics(pred_mask, gt_mask, 34)
        fig.suptitle(f"Accuracy: {metrics['accuracy']:.4f} | IoU: {metrics['mean_iou']:.4f}", 
                    fontsize=16, fontweight='bold')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(image)
        axes[0].set_title('Input Image', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(pred_colored)
        axes[1].set_title('Prediction', fontsize=14)
        axes[1].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()


def test_on_dataset(model, test_loader, device, n_classes, save_dir: Path = None):
    """
    Test model on entire test set
    
    Args:
        model: Trained model
        test_loader: Test dataloader
        device: Device
        n_classes: Number of classes
        save_dir: Directory to save visualizations (optional)
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    logger.info("Running inference on test set...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device).long()
            
            # Predict
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            # Collect predictions
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
            
            # Save first few visualizations
            if save_dir and batch_idx < 10:
                for i in range(min(4, len(images))):
                    img = images[i].cpu().numpy().transpose(1, 2, 0)
                    # Denormalize
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img = img * std + mean
                    img = np.clip(img, 0, 1)
                    img = (img * 255).astype(np.uint8)
                    
                    pred = preds[i].cpu().numpy()
                    gt = masks[i].cpu().numpy()
                    
                    # Create visualization
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    axes[0].imshow(img)
                    axes[0].set_title('Input Image')
                    axes[0].axis('off')
                    
                    axes[1].imshow(colorize_mask(gt, COLOR_MAP))
                    axes[1].set_title('Ground Truth')
                    axes[1].axis('off')
                    
                    axes[2].imshow(colorize_mask(pred, COLOR_MAP))
                    axes[2].set_title('Prediction')
                    axes[2].axis('off')
                    
                    metrics = calculate_metrics(pred, gt, n_classes)
                    fig.suptitle(f"Acc: {metrics['accuracy']:.4f} | IoU: {metrics['mean_iou']:.4f}")
                    
                    save_path = save_dir / f'test_{batch_idx}_{i}.png'
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
    
    # Concatenate all predictions
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate overall metrics
    total_metrics = calculate_metrics(
        all_preds.flatten(), 
        all_targets.flatten(), 
        n_classes
    )
    
    return total_metrics


def main():
    """Main inference function"""
    
    # Configuration
    CHECKPOINT_PATH = 'models/checkpoints/best_model.pth'  # Change as needed
    TEST_SINGLE_IMAGE = False  # Set to True to test single image
    SINGLE_IMAGE_PATH = 'data/processed/images/10004.png'  # Change as needed
    OUTPUT_DIR = Path('inference_results')
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    if not Path(CHECKPOINT_PATH).exists():
        logger.error(f"Checkpoint not found: {CHECKPOINT_PATH}")
        logger.info("Please train the model first using: python train.py")
        return
    
    model, config = load_model(CHECKPOINT_PATH, device)
    
    # Test single image
    if TEST_SINGLE_IMAGE:
        logger.info("="*80)
        logger.info("TESTING SINGLE IMAGE")
        logger.info("="*80)
        
        if not Path(SINGLE_IMAGE_PATH).exists():
            logger.error(f"Image not found: {SINGLE_IMAGE_PATH}")
            return
        
        logger.info(f"Processing: {SINGLE_IMAGE_PATH}")
        
        # Predict
        pred_mask = predict_single_image(
            model, SINGLE_IMAGE_PATH, device, config['img_size']
        )
        
        # Visualize
        gt_mask_path = SINGLE_IMAGE_PATH.replace('/images/', '/annotations/')
        output_path = OUTPUT_DIR / f"prediction_{Path(SINGLE_IMAGE_PATH).stem}.png"
        
        visualize_prediction(
            SINGLE_IMAGE_PATH, pred_mask, gt_mask_path, str(output_path)
        )
        
        logger.info(f"Results saved to {output_path}")
    
    # Test on full test set
    else:
        logger.info("="*80)
        logger.info("TESTING ON TEST SET")
        logger.info("="*80)
        
        # Create test dataloader
        logger.info("Creating test dataloader...")
        _, _, test_loader = create_dataloaders(
            images_dir=config['images_dir'],
            masks_dir=config['masks_dir'],
            batch_size=config['batch_size'],
            num_workers=0,
            image_size=config['img_size'],
            num_classes=config['n_classes']
        )
        
        logger.info(f"Test batches: {len(test_loader)}")
        
        # Test
        vis_dir = OUTPUT_DIR / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        metrics = test_on_dataset(
            model, test_loader, device, config['n_classes'], vis_dir
        )
        
        # Print results
        logger.info("="*80)
        logger.info("TEST RESULTS")
        logger.info("="*80)
        logger.info(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Mean IoU: {metrics['mean_iou']:.4f}")
        
        # Save metrics
        metrics_path = OUTPUT_DIR / 'test_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({
                'accuracy': float(metrics['accuracy']),
                'mean_iou': float(metrics['mean_iou']),
                'per_class_iou': [float(x) if not np.isnan(x) else None 
                                 for x in metrics['per_class_iou']]
            }, f, indent=4)
        
        logger.info(f"Metrics saved to {metrics_path}")
        logger.info(f"Visualizations saved to {vis_dir}")
        
        logger.info("="*80)
        logger.info("TESTING COMPLETE!")
        logger.info("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nTesting interrupted by user")
    except Exception as e:
        logger.error(f"\n\nTesting failed: {e}", exc_info=True)
