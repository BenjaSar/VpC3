#!/usr/bin/env python3
"""
Inference Script: Test Trained Floor Plan Segmentation Model
Generates predictions, visualizations, and metrics for test images
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import json
import argparse
from tqdm import tqdm
from datetime import datetime

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


def load_checkpoint(checkpoint_path, model, device):
    """Load trained model from checkpoint"""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info("✓ Model loaded successfully")
    return model


def mask_to_colored_image(mask, colors=COLORS):
    """Convert segmentation mask to colored RGB image"""
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls in range(len(colors)):
        colored[mask == cls] = colors[cls]
    return colored


def calculate_iou(pred, true, num_classes=12):
    """Calculate IoU for each class"""
    iou_per_class = []
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        true_mask = (true == cls)
        intersection = (pred_mask & true_mask).sum()
        union = (pred_mask | true_mask).sum()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0 if pred_mask.sum() == 0 and true_mask.sum() == 0 else 0.0
        
        iou_per_class.append(float(iou))
    
    return iou_per_class


def calculate_metrics(pred, true, num_classes=12):
    """Calculate comprehensive metrics"""
    metrics = {}
    
    # Per-class IoU
    iou_per_class = calculate_iou(pred, true, num_classes)
    metrics['iou_per_class'] = iou_per_class
    metrics['mean_iou'] = np.mean([iou for iou in iou_per_class if iou > 0])
    
    # Pixel accuracy
    metrics['pixel_accuracy'] = np.mean(pred == true)
    
    # Per-class accuracy
    class_accuracy = []
    for cls in range(num_classes):
        mask = true == cls
        if mask.sum() > 0:
            acc = (pred[mask] == true[mask]).sum() / mask.sum()
            class_accuracy.append(float(acc))
    
    if class_accuracy:
        metrics['mean_class_accuracy'] = np.mean(class_accuracy)
    
    # Frequency weighted IoU
    weights = []
    for cls in range(num_classes):
        weights.append((true == cls).sum())
    total_pixels = sum(weights)
    
    if total_pixels > 0:
        weights = np.array(weights) / total_pixels
        metrics['fw_iou'] = np.sum([iou * w for iou, w in zip(iou_per_class, weights)])
    
    return metrics


def infer_single_image(model, image_path, device, num_classes=12, img_size=512):
    """Run inference on a single image"""
    logger.info(f"Inferring on: {image_path}")
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((img_size, img_size))
    
    # Normalize
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_array - mean) / std
    
    # To tensor and batch (ensure float32)
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).cpu().numpy()[0]
    
    return pred, np.array(img_resized)


def infer_dataset(model, dataloader, device, num_classes=12, max_samples=None):
    """Run inference on entire dataset"""
    model.eval()
    
    results = []
    all_metrics = {
        'mean_iou': [],
        'pixel_accuracy': [],
        'fw_iou': []
    }
    
    samples_processed = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inferring")):
            if max_samples and samples_processed >= max_samples:
                break
            
            images = batch['image'].to(device)
            masks_true = batch['mask'].cpu().numpy()
            filenames = batch['filename']
            
            # Inference
            outputs = model(images)
            masks_pred = outputs.argmax(dim=1).cpu().numpy()
            
            # Calculate metrics for each sample
            for i in range(len(images)):
                metrics = calculate_metrics(masks_pred[i], masks_true[i], num_classes)
                
                results.append({
                    'filename': filenames[i],
                    'metrics': metrics
                })
                
                all_metrics['mean_iou'].append(metrics['mean_iou'])
                all_metrics['pixel_accuracy'].append(metrics['pixel_accuracy'])
                if 'fw_iou' in metrics:
                    all_metrics['fw_iou'].append(metrics['fw_iou'])
                
                samples_processed += 1
    
    # Calculate dataset-level metrics
    dataset_metrics = {
        'mean_iou': np.mean(all_metrics['mean_iou']) if all_metrics['mean_iou'] else 0.0,
        'pixel_accuracy': np.mean(all_metrics['pixel_accuracy']) if all_metrics['pixel_accuracy'] else 0.0,
        'fw_iou': np.mean(all_metrics['fw_iou']) if all_metrics['fw_iou'] else 0.0,
        'num_samples': samples_processed
    }
    
    return results, dataset_metrics


def visualize_results(image, pred_mask, true_mask, filename, output_dir):
    """Create side-by-side visualization"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert masks to colored images
    pred_colored = mask_to_colored_image(pred_mask)
    true_colored = mask_to_colored_image(true_mask)
    
    # Normalize image
    image = np.clip(image / 255.0, 0, 1)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title(f"Image: {filename}")
    axes[0].axis('off')
    
    axes[1].imshow(true_colored)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    
    axes[2].imshow(pred_colored)
    axes[2].set_title("Prediction")
    axes[2].axis('off')
    
    # Add legend
    legend_elements = [mpatches.Patch(facecolor=np.array(COLORS[i])/255, 
                                      label=CLASS_NAMES[i]) 
                       for i in range(12)]
    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, -0.02), ncol=6, fontsize=8)
    
    plt.tight_layout()
    
    output_path = output_dir / f"{Path(filename).stem}_inference.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description='Inference on Floor Plan Segmentation Model')
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', action='store_true',
                       help='Run inference on entire test dataset')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single test image')
    parser.add_argument('--output', type=str, default='outputs/inference_results',
                       help='Output directory for results')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to process')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Save visualizations')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("INFERENCE: Floor Plan Segmentation Model")
    print("=" * 80)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    logger.info("Creating model...")
    model = ViTSegmentation(
        img_size=512,
        patch_size=32,
        in_channels=3,
        n_classes=12,
        embed_dim=384,
        n_encoder_layers=12,
        n_decoder_layers=3,
        n_heads=6,
        mlp_ratio=4.0,
        dropout=0.1
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Available checkpoints:")
        for cp in Path('models/checkpoints').glob('*.pth'):
            logger.error(f"  - {cp}")
        return
    
    model = load_checkpoint(checkpoint_path, model, device)
    
    # Run inference
    if args.dataset:
        logger.info("\n" + "=" * 80)
        logger.info("DATASET INFERENCE")
        logger.info("=" * 80)
        
        # Create dataloader
        _, _, test_loader = create_dataloaders(
            images_dir='data/processed/images',
            masks_dir='data/processed/annotations',
            batch_size=4,
            num_workers=0,
            image_size=512,
            num_classes=12
        )
        
        # Run inference
        results, dataset_metrics = infer_dataset(
            model, test_loader, device, num_classes=12,
            max_samples=args.max_samples
        )
        
        # Log metrics
        logger.info("\nDataset Metrics:")
        logger.info(f"  Mean IoU: {dataset_metrics['mean_iou']:.4f}")
        logger.info(f"  Pixel Accuracy: {dataset_metrics['pixel_accuracy']:.4f}")
        logger.info(f"  FW IoU: {dataset_metrics['fw_iou']:.4f}")
        logger.info(f"  Samples: {dataset_metrics['num_samples']}")
        
        # Save metrics
        metrics_path = output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({
                'dataset_metrics': dataset_metrics,
                'timestamp': datetime.now().isoformat(),
                'checkpoint': str(checkpoint_path)
            }, f, indent=4)
        logger.info(f"\n✓ Metrics saved to: {metrics_path}")
        
        # Save results
        results_path = output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"✓ Results saved to: {results_path}")
        
    elif args.image:
        logger.info("\n" + "=" * 80)
        logger.info("SINGLE IMAGE INFERENCE")
        logger.info("=" * 80)
        
        pred_mask, image = infer_single_image(model, args.image, device)
        
        # Save prediction
        pred_colored = mask_to_colored_image(pred_mask)
        pred_img = Image.fromarray(pred_colored)
        pred_path = output_dir / f"{Path(args.image).stem}_prediction.png"
        pred_img.save(pred_path)
        
        logger.info(f"✓ Prediction saved to: {pred_path}")
        
        # Show class distribution
        unique, counts = np.unique(pred_mask, return_counts=True)
        logger.info("\nPredicted class distribution:")
        for cls, count in zip(unique, counts):
            pct = count / pred_mask.size * 100
            logger.info(f"  Class {int(cls):2d} ({CLASS_NAMES[int(cls)]:15s}): {count:7d} ({pct:5.2f}%)")
    
    else:
        logger.info("\nNo mode specified. Use --dataset or --image")
        logger.info("Examples:")
        logger.info("  python scripts/inference.py --dataset")
        logger.info("  python scripts/inference.py --image test_image.png")
    
    logger.info("\n" + "=" * 80)
    logger.info("INFERENCE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
