"""Visualization utilities."""
import matplotlib.pyplot as plt
import numpy as np


def visualize_prediction(image, mask, prediction, class_names=None):
    """Visualize image, ground truth, and prediction."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='tab20')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(prediction, cmap='tab20')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def plot_training_history(history):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training History')
    ax1.legend()
    
    if 'train_iou' in history:
        ax2.plot(history['train_iou'], label='Train IoU')
        ax2.plot(history['val_iou'], label='Val IoU')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('IoU')
        ax2.set_title('IoU History')
        ax2.legend()
    
    plt.tight_layout()
    return fig
