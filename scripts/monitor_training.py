#!/usr/bin/env python3
"""
Real-Time Training Monitor
Displays live metrics and plots during training
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
import argparse

def monitor_training(checkpoint_dir='models/checkpoints', update_interval=10):
    """
    Monitor training in real-time
    
    Args:
        checkpoint_dir: Directory containing checkpoints and history
        update_interval: Check for updates every N seconds
    """
    
    checkpoint_path = Path(checkpoint_dir)
    history_file = checkpoint_path / 'training_history.json'
    config_file = checkpoint_path / 'config.json'
    output_dir = Path('outputs/training_monitor')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("REAL-TIME TRAINING MONITOR")
    print("=" * 80)
    print(f"Monitoring: {checkpoint_dir}")
    print(f"Output: {output_dir}")
    print(f"Update interval: {update_interval}s")
    print("\nWaiting for training to start...")
    
    last_epoch = -1
    
    while True:
        try:
            # Check if training file exists
            if not history_file.exists():
                time.sleep(update_interval)
                continue
            
            # Load history
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            # Load config if available
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
            
            num_epochs = len(history.get('train_loss', []))
            
            # Check if new data
            if num_epochs <= last_epoch:
                time.sleep(update_interval)
                continue
            
            # Print progress
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Epoch {num_epochs} completed")
            
            if num_epochs > 0:
                epoch_idx = num_epochs - 1
                print(f"  Train Loss: {history['train_loss'][epoch_idx]:.4f}")
                print(f"  Val Loss:   {history['val_loss'][epoch_idx]:.4f}")
                print(f"  Train IoU:  {history['train_iou'][epoch_idx]:.4f}")
                print(f"  Val IoU:    {history['val_iou'][epoch_idx]:.4f}")
                print(f"  Active Classes: {history['active_classes'][epoch_idx]}/12")
                print(f"  Learning Rate: {history['lr'][epoch_idx]:.6f}")
            
            # Generate plots every 5 epochs
            if num_epochs % 5 == 0 and num_epochs > 0:
                print("  → Generating plots...")
                generate_plots(history, output_dir)
            
            last_epoch = num_epochs
            time.sleep(update_interval)
        
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(update_interval)


def generate_plots(history, output_dir):
    """Generate and save monitoring plots"""
    
    epochs = np.arange(1, len(history['train_loss']) + 1)
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Loss curve
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train', linewidth=2, marker='o', markersize=3)
    ax.plot(epochs, history['val_loss'], label='Validation', linewidth=2, marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: IoU curve
    ax = axes[0, 1]
    ax.plot(epochs, history['train_iou'], label='Train', linewidth=2, marker='o', markersize=3)
    ax.plot(epochs, history['val_iou'], label='Validation', linewidth=2, marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IoU')
    ax.set_title('Intersection over Union (IoU)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Active classes
    ax = axes[1, 0]
    ax.plot(epochs, history['active_classes'], color='purple', linewidth=2, marker='o', markersize=4)
    ax.axhline(y=12, color='red', linestyle='--', label='Target (12 classes)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Active Classes')
    ax.set_title('Number of Learned Classes')
    ax.set_ylim(0, 13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Learning rate
    ax = axes[1, 1]
    ax.plot(epochs, history['lr'], color='orange', linewidth=2, marker='D', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'training_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Generate summary text
    if len(history['train_loss']) > 0:
        summary_path = output_dir / 'training_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"TRAINING SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            epoch = len(history['train_loss'])
            f.write(f"Current Epoch: {epoch}\n\n")
            
            # Best metrics
            best_val_iou_idx = np.argmax(history['val_iou'])
            best_val_loss_idx = np.argmin(history['val_loss'])
            best_active_idx = np.argmax(history['active_classes'])
            
            f.write("BEST METRICS:\n")
            f.write(f"  Best Val IoU: {history['val_iou'][best_val_iou_idx]:.4f} (Epoch {best_val_iou_idx+1})\n")
            f.write(f"  Best Val Loss: {history['val_loss'][best_val_loss_idx]:.4f} (Epoch {best_val_loss_idx+1})\n")
            f.write(f"  Most Active Classes: {history['active_classes'][best_active_idx]}/12 (Epoch {best_active_idx+1})\n\n")
            
            # Current metrics
            f.write("CURRENT METRICS (Latest Epoch):\n")
            f.write(f"  Train Loss: {history['train_loss'][-1]:.4f}\n")
            f.write(f"  Val Loss: {history['val_loss'][-1]:.4f}\n")
            f.write(f"  Train IoU: {history['train_iou'][-1]:.4f}\n")
            f.write(f"  Val IoU: {history['val_iou'][-1]:.4f}\n")
            f.write(f"  Active Classes: {history['active_classes'][-1]}/12\n")
            f.write(f"  Learning Rate: {history['lr'][-1]:.6f}\n\n")
            
            # Trends
            if epoch > 1:
                loss_trend = history['val_loss'][-1] - history['val_loss'][-2]
                iou_trend = history['val_iou'][-1] - history['val_iou'][-2]
                f.write("TRENDS (Last Epoch Change):\n")
                f.write(f"  Val Loss: {loss_trend:+.4f} {'↓ Improving' if loss_trend < 0 else '↑ Degrading'}\n")
                f.write(f"  Val IoU: {iou_trend:+.4f} {'↑ Improving' if iou_trend > 0 else '↓ Degrading'}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time training monitor')
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints',
                       help='Directory containing training checkpoints')
    parser.add_argument('--interval', type=int, default=10,
                       help='Update interval in seconds')
    
    args = parser.parse_args()
    
    monitor_training(checkpoint_dir=args.checkpoint_dir, update_interval=args.interval)
