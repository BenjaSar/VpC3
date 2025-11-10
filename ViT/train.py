#!/home/ubuntu/floorplan_classifier/VpC3/floorplan_vit/bin/python3
"""
Enhanced Training Script for ViT Floor Plan Segmentation
Incorporates improvements for severe class imbalance:
- Focal Loss for hard example mining
- Dice Loss for equal class treatment
- Improved log-scaled class weighting
- Multi-stage learning rate scheduling
- Better gradient monitoring
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import json
import argparse
from datetime import datetime
from collections import Counter

# Import project modules
from data.dataset import create_dataloaders
from src.utils.logging_config import setup_logging

logger = setup_logging()


# ==================== Model Architecture ====================

class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings"""
    def __init__(self, img_size=512, patch_size=16, in_channels=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        """x: (B, 3, H, W) -> (B, n_patches, embed_dim)"""
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class TransformerEncoderLayer(nn.Module):
    """Standard transformer encoder layer with self-attention and MLP"""
    def __init__(self, embed_dim=384, n_heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderLayer(nn.Module):
    """Transformer decoder layer"""
    def __init__(self, embed_dim=384, n_heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class SegmentationHead(nn.Module):
    """Enhanced segmentation head with deeper decoder"""
    def __init__(self, embed_dim=384, patch_size=16, img_size=512, n_classes=256):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.n_patches_side = img_size // patch_size
        
        # Progressive upsampling with batch norm and dropout
        self.conv_transpose = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(16, n_classes, kernel_size=4, stride=2, padding=1)
        )
        
    def forward(self, x):
        B = x.shape[0]
        x = x.transpose(1, 2)
        x = x.reshape(B, -1, self.n_patches_side, self.n_patches_side)
        x = self.conv_transpose(x)
        return x


class ViTSegmentation(nn.Module):
    """Vision Transformer for semantic segmentation"""
    def __init__(self, img_size=512, patch_size=16, in_channels=3, n_classes=256,
                 embed_dim=384, n_encoder_layers=12, n_decoder_layers=4, 
                 n_heads=6, mlp_ratio=4.0, dropout=0.2):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # Transformer decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_decoder_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.seg_head = SegmentationHead(embed_dim, patch_size, img_size, n_classes)
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        # Encoder
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Decoder
        for layer in self.decoder_layers:
            x = layer(x)
        
        # Segmentation head
        x = self.norm(x)
        x = self.seg_head(x)
        
        return x


# ==================== Loss Functions ====================

class FocalLoss(nn.Module):
    """
    Focal Loss: L = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Focuses on hard examples, reduces impact of easy negatives.
    Great for class imbalance.
    
    Args:
        alpha: Class weights (e.g., inverse frequency)
        gamma: Focusing parameter (higher = more focus on hard examples)
        ignore_index: Class index to ignore
    """
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
    def forward(self, inputs, targets):
        # Get cross entropy loss
        ce_loss = nn.functional.cross_entropy(
            inputs, targets,
            weight=self.alpha,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        
        # Get softmax probabilities
        p = torch.softmax(inputs, dim=1)
        
        # Get the probability of the true class
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Calculate focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma
        
        # Combine: focal weight * cross entropy
        focal_loss = focal_weight * ce_loss
        
        # Mask out ignore_index
        mask = targets != self.ignore_index
        focal_loss = focal_loss[mask]
        
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """
    Dice Loss: L = 1 - (2 * intersection / union)
    
    Better for class imbalance, treats all classes equally.
    Complements cross-entropy by focusing on IoU.
    
    Args:
        ignore_index: Class index to ignore
        smooth: Smoothing constant to avoid division by zero
    """
    def __init__(self, ignore_index=255, smooth=1e-5):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # inputs: (B, C, H, W)
        # targets: (B, H, W)
        
        probs = torch.softmax(inputs, dim=1)
        
        dice_losses = []
        for cls in range(inputs.shape[1]):
            if cls == self.ignore_index:
                continue
            
            # Get probability for this class
            pred_cls = probs[:, cls, :, :]
            
            # Get target mask for this class
            target_cls = (targets == cls).float()
            
            # Calculate intersection and union
            intersection = (pred_cls * target_cls).sum()
            dice = (2.0 * intersection + self.smooth) / \
                   (pred_cls.sum() + target_cls.sum() + self.smooth)
            
            # Append 1 - dice (loss increases when dice decreases)
            dice_losses.append(1.0 - dice)
        
        return torch.stack(dice_losses).mean() if dice_losses else torch.tensor(0.0, device=inputs.device)


class CombinedLoss(nn.Module):
    """
    Combines multiple losses for better convergence:
    - Focal Loss: Focuses on hard examples
    - Dice Loss: Treats all classes equally
    - Cross Entropy: Standard baseline
    """
    def __init__(self, weight=None, ignore_index=255, loss_type='hybrid'):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_type = loss_type
        
        if loss_type in ['hybrid', 'focal+ce']:
            self.focal_loss = FocalLoss(alpha=weight, gamma=2.0, ignore_index=ignore_index)
        
        if loss_type in ['hybrid', 'dice+ce']:
            self.dice_loss = DiceLoss(ignore_index=ignore_index)
        
        if loss_type in ['hybrid', 'ce', 'focal+ce', 'dice+ce']:
            self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    
    def forward(self, inputs, targets):
        if self.loss_type == 'ce':
            return self.ce_loss(inputs, targets)
        
        elif self.loss_type == 'focal':
            return self.focal_loss(inputs, targets)
        
        elif self.loss_type == 'dice':
            return self.dice_loss(inputs, targets)
        
        elif self.loss_type == 'focal+ce':
            # 0.6 focal + 0.4 cross entropy
            focal = self.focal_loss(inputs, targets)
            ce = self.ce_loss(inputs, targets)
            return 0.6 * focal + 0.4 * ce
        
        elif self.loss_type == 'dice+ce':
            # 0.5 dice + 0.5 cross entropy
            dice = self.dice_loss(inputs, targets)
            ce = self.ce_loss(inputs, targets)
            return 0.5 * dice + 0.5 * ce
        
        elif self.loss_type == 'hybrid':
            # All three: 0.4 focal + 0.3 dice + 0.3 ce
            focal = self.focal_loss(inputs, targets)
            dice = self.dice_loss(inputs, targets)
            ce = self.ce_loss(inputs, targets)
            return 0.4 * focal + 0.3 * dice + 0.3 * ce
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


# ==================== Utility Classes ====================

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=20, min_delta=0.001, mode='max', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self._is_improvement(score):
            if self.verbose:
                improvement = score - self.best_score if self.mode == 'max' else self.best_score - score
                logger.info(f"    ✓ Validation improved by {improvement:.4f} (new best: {score:.4f})")
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose and self.counter > 0:
                logger.info(f"    ⏱ Early stopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
    
    def _is_improvement(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


# ==================== Loss and Weighting ====================

def calculate_class_weights_improved(dataloader, num_classes=256, device='cpu'):
    """
    Calculate class weights using log-scaled inverse frequency.
    
    This approach:
    1. Uses log scaling to handle extreme imbalance
    2. Normalizes by median to prevent extreme ratios
    3. Bounds weights to reasonable range
    
    Much better than simple inverse frequency for 256 classes!
    """
    logger.info("Calculating improved class weights (log-scaled)...")
    
    class_counts = Counter()
    total_pixels = 0
    
    # Count pixels per class
    for batch in tqdm(dataloader, desc="Analyzing class distribution"):
        masks = batch['mask'].numpy()
        for mask in masks:
            unique, counts = np.unique(mask, return_counts=True)
            for cls, count in zip(unique, counts):
                cls_int = int(cls)
                if cls_int != 255:
                    class_counts[cls_int] += int(count)
                    total_pixels += int(count)
    
    # Calculate weights with log scaling
    weights = []
    for i in range(num_classes):
        if i == 255:
            weights.append(0.0)
        else:
            count = class_counts.get(i, 1)
            
            # Log-scaled inverse frequency
            # Handles extreme imbalance better than raw inverse frequency
            if count > 0:
                # Calculate base ratio
                ratio = total_pixels / ((num_classes - 1) * count)
                
                # Use log to compress extreme values
                log_weight = np.log1p(ratio)
                max_log_weight = np.log1p(total_pixels / (num_classes - 1))
                
                # Normalize to [0, 1]
                normalized = log_weight / max_log_weight
                
                # Scale to [0.3, 3.0] for better training dynamics
                weight = 0.3 + (normalized * 2.7)
            else:
                weight = 3.0
            
            weights.append(weight)
    
    weights = torch.tensor(weights, dtype=torch.float32)
    
    # Normalize by median of valid weights
    valid_weights = weights[:255]
    median_weight = torch.median(valid_weights)
    weights = weights / (median_weight + 1e-8)
    
    # Log statistics
    logger.info("\n" + "="*70)
    logger.info("IMPROVED CLASS WEIGHTS (Log-scaled)")
    logger.info("="*70)
    logger.info(f"  Min weight: {weights.min():.4f}")
    logger.info(f"  Max weight: {weights.max():.4f}")
    logger.info(f"  Mean weight: {weights.mean():.4f}")
    logger.info(f"  Median weight: {torch.median(weights):.4f}")
    logger.info(f"  Weight ratio (max/min): {(weights.max() / weights.min()):.2f}x")
    
    # Show weight distribution for important classes
    logger.info("\nClass weights (top 10 by frequency):")
    top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for cls, count in top_classes:
        pct = (count / total_pixels * 100)
        logger.info(f"  Class {cls:3d}: weight={weights[cls]:.3f}, pixels={count:>8,} ({pct:>5.2f}%)")
    logger.info("="*70 + "\n")
    
    return weights.to(device)


# ==================== Metrics ====================

def calculate_iou(pred, target, n_classes, ignore_index=255):
    """Calculate mean IoU, excluding ignore_index class"""
    pred = pred.cpu().view(-1)
    target = target.cpu().view(-1)
    
    valid_mask = target != ignore_index
    pred = pred[valid_mask]
    target = target[valid_mask]
    
    ious = []
    for cls in range(n_classes):
        if cls == ignore_index:
            continue
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())
    
    return np.nanmean(ious) if len(ious) > 0 else 0.0


def calculate_per_class_iou(pred, target, n_classes, ignore_index=255):
    """Calculate per-class IoU and active classes"""
    pred = pred.cpu().view(-1)
    target = target.cpu().view(-1)
    
    valid_mask = target != ignore_index
    pred = pred[valid_mask]
    target = target[valid_mask]
    
    per_class_iou = {}
    for cls in range(n_classes):
        if cls == ignore_index:
            continue
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            per_class_iou[cls] = 0.0
        else:
            per_class_iou[cls] = (intersection / union).item()
    
    active_classes = sum(1 for iou in per_class_iou.values() if iou > 0.01)
    
    return per_class_iou, active_classes


# ==================== Training and Validation ====================

def train_epoch(model, dataloader, criterion, optimizer, device, n_classes, scaler=None, 
                accumulation_steps=1):
    """Train for one epoch with gradient accumulation support"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device).long()
        masks = torch.clamp(masks, 0, n_classes - 1)
        
        # Mixed precision training
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss = loss / accumulation_steps  # Scale loss for accumulation
            
            scaler.scale(loss).backward()
            
            # Step optimizer every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss = loss / accumulation_steps
            
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # Collect predictions
        pred = outputs.detach().argmax(dim=1)
        all_preds.append(pred.cpu())
        all_targets.append(masks.cpu())
        
        total_loss += loss.item() * accumulation_steps
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Final optimizer step if needed
    if (batch_idx + 1) % accumulation_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    avg_iou = calculate_iou(all_preds.view(-1), all_targets.view(-1), n_classes)
    per_class_iou, active_classes = calculate_per_class_iou(
        all_preds.view(-1), all_targets.view(-1), n_classes
    )
    
    return avg_loss, avg_iou, per_class_iou, active_classes


def validate(model, dataloader, criterion, device, n_classes):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device).long()
            masks = torch.clamp(masks, 0, n_classes - 1)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            pred = outputs.argmax(dim=1)
            all_preds.append(pred.cpu())
            all_targets.append(masks.cpu())
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    avg_iou = calculate_iou(all_preds.view(-1), all_targets.view(-1), n_classes)
    per_class_iou, active_classes = calculate_per_class_iou(
        all_preds.view(-1), all_targets.view(-1), n_classes
    )
    
    return avg_loss, avg_iou, per_class_iou, active_classes


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load checkpoint and restore training state"""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("[✓] Model state loaded")
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    logger.info("[✓] Optimizer state loaded")
    
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    logger.info("[✓] Scheduler state loaded")
    
    start_epoch = checkpoint['epoch'] + 1
    history = checkpoint.get('history', {
        'train_loss': [],
        'train_iou': [],
        'val_loss': [],
        'val_iou': [],
        'active_classes': [],
        'lr': []
    })
    
    best_val_iou = max(history.get('val_iou', [0.0]))
    best_active_classes = max(history.get('active_classes', [0]))
    
    logger.info(f"[✓] Resuming from epoch {start_epoch}")
    logger.info(f"    Previous best IoU: {best_val_iou:.4f}")
    
    return start_epoch, history, best_val_iou, best_active_classes


def save_checkpoint(checkpoint_path, epoch, model, optimizer, scheduler,
                   val_loss, val_iou, active_classes, history, config, tag=""):
    """Save checkpoint to disk"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'val_iou': val_iou,
        'active_classes': active_classes,
        'history': history,
        'config': config
    }, checkpoint_path)
    
    tag_str = f" ({tag})" if tag else ""
    logger.info(f"[✓] Checkpoint saved{tag_str}")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced ViT Floor Plan Segmentation with Focal+Dice Loss'
    )
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size (default: 8 for better balance)')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Total epochs')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate (default: 5e-5, lower than before)')
    parser.add_argument('--patch-size', type=int, default=16,
                       help='Vision Transformer patch size (default: 16 for more patches)')
    parser.add_argument('--loss', type=str, default='hybrid',
                       choices=['ce', 'focal', 'dice', 'focal+ce', 'dice+ce', 'hybrid'],
                       help='Loss function type (default: hybrid)')
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints_v2_improved',
                       help='Checkpoint directory')
    parser.add_argument('--no-early-stop', action='store_true',
                       help='Disable early stopping')
    parser.add_argument('--accumulation-steps', type=int, default=2,
                       help='Gradient accumulation steps (effective batch size = batch_size * accumulation_steps)')
    
    args = parser.parse_args()
    
    # Configuration with improvements
    CONFIG = {
        # Data
        'images_dir': 'data/processed/images',
        'masks_dir': 'data/processed/annotations',
        'batch_size': args.batch_size,
        'num_workers': 0,
        
        # Model (smaller patches for more detail)
        'img_size': 512,
        'patch_size': args.patch_size,
        'n_classes': 256,
        'embed_dim': 384,
        'n_encoder_layers': 12,
        'n_decoder_layers': 4,  # Deeper decoder
        'n_heads': 6,
        'mlp_ratio': 4.0,
        'dropout': 0.2,  # Higher dropout for regularization
        
        # Training (conservative for stability)
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': 0.001,
        'mixed_precision': True,
        'gradient_accumulation_steps': args.accumulation_steps,
        
        # Loss function (improved for imbalance)
        'loss_type': args.loss,
        'use_class_weights': True,
        'label_smoothing': 0.15,
        
        # Early stopping (more patient)
        'early_stopping': {
            'enabled': not args.no_early_stop,
            'patience': 30,
            'min_delta': 0.0001,
            'monitor': 'val_iou',
            'mode': 'max',
        },
        
        # Checkpointing
        'checkpoint_dir': args.checkpoint_dir,
        'save_frequency': 10
    }
    
    # Create checkpoint directory
    checkpoint_dir = Path(CONFIG['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = checkpoint_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=4)
    logger.info(f"Configuration saved to {config_path}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\nUsing device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create dataloaders
    logger.info("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        images_dir=CONFIG['images_dir'],
        masks_dir=CONFIG['masks_dir'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        image_size=CONFIG['img_size'],
        num_classes=CONFIG['n_classes']
    )
    
    logger.info(f"Train batches: {len(train_loader)} (effective batch: {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']})")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Create model
    logger.info("\nCreating model...")
    model = ViTSegmentation(
        img_size=CONFIG['img_size'],
        patch_size=CONFIG['patch_size'],
        in_channels=3,
        n_classes=CONFIG['n_classes'],
        embed_dim=CONFIG['embed_dim'],
        n_encoder_layers=CONFIG['n_encoder_layers'],
        n_decoder_layers=CONFIG['n_decoder_layers'],
        n_heads=CONFIG['n_heads'],
        mlp_ratio=CONFIG['mlp_ratio'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params / 1e6:.2f}M")
    
    # Calculate class weights and create loss
    logger.info("\nPreparing loss function...")
    class_weights = calculate_class_weights_improved(train_loader, CONFIG['n_classes'], device)
    
    criterion = CombinedLoss(
        weight=class_weights,
        ignore_index=255,
        loss_type=CONFIG['loss_type']
    )
    logger.info(f"[✓] Using {CONFIG['loss_type'].upper()} loss")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Multi-stage learning rate scheduler
    warmup_epochs = 5
    main_epochs = 40
    finetune_epochs = CONFIG['num_epochs'] - warmup_epochs - main_epochs
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.001, end_factor=1.0,
                total_iters=warmup_epochs * len(train_loader)
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=main_epochs, eta_min=1e-6
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.99
            )
        ],
        milestones=[warmup_epochs * len(train_loader), 
                   (warmup_epochs + main_epochs) * len(train_loader)]
    )
    
    logger.info("[✓] Using multi-stage learning rate scheduler")
    logger.info(f"    Stage 1 (Warmup): {warmup_epochs} epochs")
    logger.info(f"    Stage 2 (Main): {main_epochs} epochs")
    logger.info(f"    Stage 3 (Finetune): {finetune_epochs} epochs")
    
    # Mixed precision
    scaler = GradScaler() if CONFIG['mixed_precision'] and torch.cuda.is_available() else None
    if scaler:
        logger.info("[✓] Mixed precision training enabled")
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_val_iou = 0.0
    history = {
        'train_loss': [],
        'train_iou': [],
        'val_loss': [],
        'val_iou': [],
        'active_classes': [],
        'lr': []
    }
    
    if args.resume and Path(args.resume).exists():
        logger.info("\n" + "="*80)
        logger.info("RESUMING FROM CHECKPOINT")
        logger.info("="*80)
        start_epoch, history, best_val_iou, _ = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
    
    # Early stopping
    early_stopping = None
    if CONFIG['early_stopping']['enabled']:
        early_stopping = EarlyStopping(
            patience=CONFIG['early_stopping']['patience'],
            min_delta=CONFIG['early_stopping']['min_delta'],
            mode=CONFIG['early_stopping']['mode']
        )
        logger.info("\n" + "="*80)
        logger.info("EARLY STOPPING ENABLED")
        logger.info(f"Patience: {CONFIG['early_stopping']['patience']} epochs")
        logger.info("="*80)
    
    # Training loop
    logger.info("\n" + "="*80)
    logger.info("STARTING ENHANCED TRAINING")
    logger.info("="*80)
    
    for epoch in range(start_epoch, CONFIG['num_epochs']):
        logger.info(f"\n📍 Epoch {epoch+1}/{CONFIG['num_epochs']}")
        logger.info("-" * 80)
        
        # Train
        train_loss, train_iou, _, train_active = train_epoch(
            model, train_loader, criterion, optimizer, device, CONFIG['n_classes'],
            scaler, CONFIG['gradient_accumulation_steps']
        )
        
        logger.info(f"Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Active: {train_active}")
        
        # Validate
        val_loss, val_iou, _, val_active = validate(
            model, val_loader, criterion, device, CONFIG['n_classes']
        )
        
        logger.info(f"Val - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Active: {val_active}")
        
        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"LR: {current_lr:.6f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['active_classes'].append(val_active)
        history['lr'].append(current_lr)
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_model_path = checkpoint_dir / 'best_model.pth'
            save_checkpoint(best_model_path, epoch, model, optimizer, scheduler,
                          val_loss, val_iou, val_active, history, CONFIG,
                          tag=f"BEST (IoU: {val_iou:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % CONFIG['save_frequency'] == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint(checkpoint_path, epoch, model, optimizer, scheduler,
                          val_loss, val_iou, val_active, history, CONFIG)
        
        # Early stopping
        if early_stopping and early_stopping(val_iou, epoch):
            logger.info("\n" + "="*80)
            logger.info("🛑 EARLY STOPPING TRIGGERED")
            logger.info("="*80)
            break
    
    # Save final model
    final_model_path = checkpoint_dir / 'final_model.pth'
    save_checkpoint(final_model_path, CONFIG['num_epochs'] - 1, model, optimizer, scheduler,
                   val_loss, val_iou, val_active, history, CONFIG, tag="FINAL")
    
    history_path = checkpoint_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETED")
    logger.info("="*80)
    logger.info(f"Best IoU: {best_val_iou:.4f}")
    logger.info(f"Models saved to: {checkpoint_dir}")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⏸ Training interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}", exc_info=True)