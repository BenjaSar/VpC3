"""
Evaluation Script for Floor Plan Segmentation
Provides comprehensive per-class metrics and visualizations
"""
import os
import sys
import yaml
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from metrics import SegmentationMetrics
from data.dataset import FloorPlanDataset, get_augmentation_pipeline
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_model(checkpoint_path, config, device):
    """Load trained model from checkpoint"""
    model = smp.UnetPlusPlus(
        encoder_name=config['model']['encoder'],
        encoder_weights=None,  # Load from checkpoint
        in_channels=3,
        classes=config['model']['num_classes'],
        activation=None,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def evaluate_model(model, dataloader, num_classes, class_names, device):
    """Evaluate model on dataset"""
    metrics = SegmentationMetrics(
        num_classes=num_classes,
        ignore_index=255,
        class_names=class_names
    )
    
    print("Evaluating model...")
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Evaluation'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            metrics.update(predictions, masks)
    
    return metrics


def plot_confusion_matrix(confusion_matrix, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(20, 18))
    
    # Normalize confusion matrix
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    # Plot
    sns.heatmap(
        cm_normalized,
        annot=False,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Frequency'}
    )
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {save_path}")


def plot_per_class_metrics(metrics_dict, class_names, save_path):
    """Plot per-class metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # IoU
    axes[0, 0].bar(range(len(class_names)), metrics_dict['per_class_iou'])
    axes[0, 0].set_title('Per-Class IoU', fontsize=14)
    axes[0, 0].set_xlabel('Class')
    axes[0, 0].set_ylabel('IoU')
    axes[0, 0].set_xticks(range(len(class_names)))
    axes[0, 0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Dice
    axes[0, 1].bar(range(len(class_names)), metrics_dict['per_class_dice'])
    axes[0, 1].set_title('Per-Class Dice Score', fontsize=14)
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].set_ylabel('Dice')
    axes[0, 1].set_xticks(range(len(class_names)))
    axes[0, 1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Precision
    axes[1, 0].bar(range(len(class_names)), metrics_dict['per_class_precision'])
    axes[1, 0].set_title('Per-Class Precision', fontsize=14)
    axes[1, 0].set_xlabel('Class')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_xticks(range(len(class_names)))
    axes[1, 0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Recall
    axes[1, 1].bar(range(len(class_names)), metrics_dict['per_class_recall'])
    axes[1, 1].set_title('Per-Class Recall', fontsize=14)
    axes[1, 1].set_xlabel('Class')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_xticks(range(len(class_names)))
    axes[1, 1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Per-class metrics plot saved to: {save_path}")


def save_metrics_report(metrics_dict, rare_metrics, class_names, save_path):
    """Save detailed metrics report to JSON"""
    report = {
        'overall_metrics': {
            'mean_iou': float(metrics_dict['mean_iou']),
            'weighted_iou': float(metrics_dict['weighted_iou']),
            'mean_dice': float(metrics_dict['mean_dice']),
            'macro_f1': float(metrics_dict['macro_f1']),
            'weighted_f1': float(metrics_dict['weighted_f1'])
        },
        'rare_class_metrics': {
            'rare_class_count': int(rare_metrics['rare_class_count']),
            'rare_class_f1': float(rare_metrics['rare_class_f1'])
        },
        'per_class_metrics': {}
    }
    
    # Add per-class metrics
    for i, class_name in enumerate(class_names):
        report['per_class_metrics'][class_name] = {
            'iou': float(metrics_dict['per_class_iou'][i]),
            'dice': float(metrics_dict['per_class_dice'][i]),
            'precision': float(metrics_dict['per_class_precision'][i]),
            'recall': float(metrics_dict['per_class_recall'][i]),
            'f1': float(metrics_dict['per_class_f1'][i])
        }
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Metrics report saved to: {save_path}")


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate floor plan segmentation model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='VpC3/configs/config_enhanced.yaml',
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='VpC3/evaluation_results',
                        help='Directory to save evaluation results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model, checkpoint = load_model(args.checkpoint, config, device)
    
    print(f"Model trained for {checkpoint['epoch']} epochs")
    print(f"Best validation IoU: {checkpoint['metrics']['mean_iou']:.4f}")
    
    # Setup validation augmentation pipeline
    val_augmentation = get_augmentation_pipeline(
        config['data']['image_size'],
        split='val'
    )
    
    # Setup dataset
    val_dataset = FloorPlanDataset(
        config['data']['val_images'],
        config['data']['val_masks'],
        image_size=config['data']['image_size'],
        augmentation=val_augmentation,
        split='val',
        num_classes=config['model']['num_classes']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    print(f"Validation samples: {len(val_dataset)}")
    
    # Get class names
    class_names = [config['class_names'][i] for i in range(config['model']['num_classes'])]
    
    # Evaluate
    metrics = evaluate_model(
        model, val_loader,
        config['model']['num_classes'],
        class_names,
        device
    )
    
    # Print metrics
    print("\n" + "="*80)
    metrics.print_metrics(detailed=True)
    
    # Get all metrics
    metrics_dict = metrics.compute_all_metrics()
    rare_metrics = metrics.get_rare_class_metrics(threshold=0.01)
    
    # Save confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(metrics_dict['confusion_matrix'], class_names, cm_path)
    
    # Save per-class metrics plot
    metrics_plot_path = os.path.join(args.output_dir, 'per_class_metrics.png')
    plot_per_class_metrics(metrics_dict, class_names, metrics_plot_path)
    
    # Save metrics report
    report_path = os.path.join(args.output_dir, 'evaluation_report.json')
    save_metrics_report(metrics_dict, rare_metrics, class_names, report_path)
    
    print("\n" + "="*80)
    print("Evaluation completed!")
    print(f"Results saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
