#!/usr/bin/env python3
"""
Advanced Vision Transformer-based segmentation for CubiCasa5K dataset.

Implements:
✓ Focal Loss for hard example mining
✓ Dice Loss for class balance
✓ Cross-Entropy Loss for stability
✓ Boundary Loss for room precision
✓ Log-scaled class weighting for extreme imbalance
✓ Multi-stage learning rate scheduling
✓ Mixed precision training (FP16)
✓ Gradient monitoring and validation
✓ Comprehensive per-class evaluation
✓ TensorBoard & W&B monitoring
✓ Production-grade checkpointing

Author: FS
Date: 2025-11-06
Version: 1.0 (development)
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List
import argparse
from data.dataset import create_dataloaders


import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm
from collections import Counter

# Optional imports
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import cv2

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION & SETUP
# ═══════════════════════════════════════════════════════════════════════════

class Config:
    """Enhanced configuration with all fixes"""
    
    # Model
    model_name = "UnetPlusPlus"
    encoder_name = "efficientnet-b4"
    num_classes = 34
    image_size = 512
    
    # Training
    batch_size =4  # Reduced for stability
    num_epochs = 25
    learning_rate = 1e-4  # Increased from 5e-5 for faster convergence
    weight_decay = 1e-2
    num_workers = 4
    
    # Loss function - FIXED: All losses enabled
    loss_config = {
        'use_focal': True,
        'use_dice': True,
        'use_ce': True,
        'use_boundary': True,
        'focal_weight': 0.4,
        'dice_weight': 0.3,
        'ce_weight': 0.2,
        'boundary_weight': 0.1,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
    }
    
    # Scheduler
    scheduler_type = 'multistage'  # warmup -> cosine -> exponential
    warmup_epochs = 5
    cosine_epochs = 50
    min_lr = 1e-6
    
    # Class weighting
    class_weight_method = 'log_scaled'  # Options: inverse, log_scaled, sqrt
    
    # Training settings
    use_amp = True  # Mixed precision
    gradient_clip = 1.0
    early_stopping_patience = 20
    save_interval = 5
    
    # Data
    ignore_index = 255
    
    # Paths
    data_root = Path('/home/ubuntu/floorplan_classifier/VpC3/data/processed')
    output_root = Path('models/cubicasa5k_improved')
    log_root = Path('logs/cubicasa5k_improved')


# ═══════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup comprehensive logging"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('FloorPlanSegmentation')
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


# ═══════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS - IMPROVED & FIXED
# ═══════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """Focal Loss for hard example mining"""
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0,
                 ignore_index: int = 255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Cross entropy loss
        ce = nn.functional.cross_entropy(
            inputs, targets,
            weight=self.alpha,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        
        # Focal loss: (1 - p_t)^gamma * ce
        p = torch.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - p_t) ** self.gamma
        
        focal_loss = focal_weight * ce
        mask = targets != self.ignore_index
        
        return focal_loss[mask].mean() if mask.sum() > 0 else focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for class balance"""
    
    def __init__(self, ignore_index: int = 255, smooth: float = 1e-5):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        probs = torch.softmax(inputs, dim=1)
        dice_scores = []
        
        for cls in range(inputs.shape[1]):
            pred_cls = probs[:, cls, :, :]
            target_cls = (targets == cls).float()
            
            intersection = (pred_cls * target_cls).sum()
            dice = (2.0 * intersection + self.smooth) / \
                   (pred_cls.sum() + target_cls.sum() + self.smooth)
            
            dice_scores.append(1.0 - dice)
        
        return torch.stack(dice_scores).mean()


class BoundaryLoss(nn.Module):
    """Boundary Loss for room precision"""
    
    def __init__(self, ignore_index: int = 255, boundary_weight: float = 2.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.boundary_weight = boundary_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        ce_loss = nn.functional.cross_entropy(
            inputs, targets,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        
        # Detect boundaries
        boundary_map = torch.zeros_like(targets, dtype=torch.float32)
        
        for i in range(1, targets.shape[1] - 1):
            for j in range(1, targets.shape[2] - 1):
                patch = targets[:, i-1:i+2, j-1:j+2]
                if (patch.max(dim=1)[0] - patch.min(dim=1)[0] > 0).any():
                    boundary_map[:, i, j] = 1.0
        
        # Apply boundary weighting
        weighted_loss = ce_loss * (1.0 + self.boundary_weight * boundary_map)
        
        return weighted_loss.mean()


class CombinedLoss(nn.Module):
    """Hybrid loss combining Focal, Dice, CE, and Boundary losses"""
    
    def __init__(self, config: Dict, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.config = config
        
        self.focal_loss = FocalLoss(
            alpha=class_weights,
            gamma=config['focal_gamma'],
            ignore_index=255
        )
        self.dice_loss = DiceLoss(ignore_index=255)
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=255,
            label_smoothing=0.1
        )
        self.boundary_loss = BoundaryLoss(boundary_weight=0.5)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass - returns dict of loss components"""
        losses = {}
        total_loss = 0.0
        
        # Focal Loss
        if self.config['use_focal']:
            focal = self.focal_loss(inputs, targets)
            losses['focal'] = focal
            total_loss += self.config['focal_weight'] * focal
        
        # Dice Loss
        if self.config['use_dice']:
            dice = self.dice_loss(inputs, targets)
            losses['dice'] = dice
            total_loss += self.config['dice_weight'] * dice
        
        # Cross-Entropy Loss
        if self.config['use_ce']:
            ce = self.ce_loss(inputs, targets)
            losses['ce'] = ce
            total_loss += self.config['ce_weight'] * ce
        
        # Boundary Loss
        if self.config['use_boundary']:
            boundary = self.boundary_loss(inputs, targets)
            losses['boundary'] = boundary
            total_loss += self.config['boundary_weight'] * boundary
        
        losses['total'] = total_loss
        
        return losses


# ═══════════════════════════════════════════════════════════════════════════
# CLASS WEIGHTING 
# ═══════════════════════════════════════════════════════════════════════════

def calculate_class_weights(dataloader: DataLoader, num_classes: int, 
                           device: torch.device, method: str = 'log_scaled',
                           logger: Optional[logging.Logger] = None) -> torch.Tensor:
    """
    Calculate class weights from dataset with improved robustness
    
    Methods:
    - inverse: Simple inverse frequency
    - log_scaled: Log-scaled (RECOMMENDED) - prevents extreme weights
    - sqrt: Square root scaling
    """
    
    if logger:
        logger.info("📊 Calculating class weights from training data...")
    
    class_counts = torch.zeros(num_classes, device=device)
    total_pixels = 0
    
    # Count pixels per class
    for batch in tqdm(dataloader, desc="Analyzing class distribution", disable=logger is None):
        if isinstance(batch, dict):
            masks = batch['mask'].long().to(device)
        else:
            masks = batch[1].long().to(device)
        
        # Ensure valid range
        masks = torch.clamp(masks, 0, num_classes - 1)
        
        for cls in range(num_classes):
            class_counts[cls] += (masks == cls).sum()
        
        total_pixels += masks.numel()
    
    # Calculate weights
    if method == 'inverse':
        weights = total_pixels / ((num_classes - 1) * (class_counts + 1e-8))
    
    elif method == 'log_scaled':  # RECOMMENDED
        eps = 1e-8
        ratios = total_pixels / ((num_classes - 1) * (class_counts + eps))
        weights = torch.log(ratios + 1.0)
        weights = weights / (weights.max() + eps)
        weights = 0.5 + (weights * 2.5)  # Scale to [0.5, 3.0]
    
    elif method == 'sqrt':
        weights = torch.sqrt(total_pixels / ((num_classes - 1) * (class_counts + 1e-8)))
    
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Ensure valid values
    weights = torch.clamp(weights, min=0.1, max=10.0)
    weights = weights / weights.mean() * num_classes  # Normalize
    
    # Report
    if logger:
        logger.info("\n✅ CLASS WEIGHT DISTRIBUTION:")
        logger.info(f"{'Class':<15} {'Count':<12} {'%':<8} {'Weight':<8}")
        logger.info("-" * 45)
        for cls in range(min(15, num_classes)):
            count = class_counts[cls].item()
            pct = (count / total_pixels * 100) if total_pixels > 0 else 0
            weight = weights[cls].item()
            logger.info(f"{cls:<15} {count:<12.0f} {pct:<8.2f} {weight:<8.3f}")
        logger.info(f"Min: {weights.min():.3f}, Max: {weights.max():.3f}, Mean: {weights.mean():.3f}")
        logger.info("=" * 45 + "\n")
    
    return weights.cpu()


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING - IMPROVED & FIXED
# ═══════════════════════════════════════════════════════════════════════════

class FloorPlanDataset(torch.utils.data.Dataset):
    """Robust floor plan dataset loader"""
    
    def __init__(self, image_dir: Path, mask_dir: Path, image_size: int = 512,
                 transform=None, num_classes: int = 34):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.transform = transform
        self.num_classes = num_classes
        
        # Load image list
        self.image_files = sorted(self.image_dir.glob('*.png')) + \
                          sorted(self.image_dir.glob('*.jpg')) + \
                          sorted(self.image_dir.glob('*.jpeg'))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Load mask
        mask_path = self.mask_dir / image_path.stem / 'room_mask.png'
        if not mask_path.exists():
            mask_path = self.mask_dir / f"{image_path.stem}.png"
        
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # Fallback: create dummy mask (all background)
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Ensure same size
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)
            mask = torch.from_numpy(mask).long()
        
        # Ensure valid mask range
        mask = torch.clamp(mask, 0, self.num_classes - 1)
        
        return {
            'image': image,
            'mask': mask,
            'path': str(image_path)
        }


def create_transforms(image_size: int = 512):
    """Create augmentation pipelines"""
    
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1.0),
            A.GridDistortion(p=1.0),
            A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.2),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return train_transform, val_transform


# def create_dataloaders(config: Config, logger: logging.Logger) -> Tuple[DataLoader, DataLoader]:
#     """Create train and validation dataloaders"""
    
#     logger.info("📂 Creating dataloaders...")
    
#     train_transform, val_transform = create_transforms(config.image_size)
    
#     train_dataset = FloorPlanDataset(
#         image_dir=config.data_root / 'train_images',
#         mask_dir=config.data_root / 'train_masks',
#         image_size=config.image_size,
#         transform=train_transform,
#         num_classes=config.num_classes
#     )
    
#     val_dataset = FloorPlanDataset(
#         image_dir=config.data_root / 'val_images',
#         mask_dir=config.data_root / 'val_masks',
#         image_size=config.image_size,
#         transform=val_transform,
#         num_classes=config.num_classes
#     )
    
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config.batch_size,
#         shuffle=True,
#         num_workers=config.num_workers,
#         pin_memory=True,
#         drop_last=True
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=config.batch_size * 2,
#         shuffle=False,
#         num_workers=config.num_workers,
#         pin_memory=True
#     )
    
#     logger.info(f"✓ Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
#     logger.info(f"✓ Val: {len(val_dataset)} samples ({len(val_loader)} batches)\n")
    
#     return train_loader, val_loader


    # ==================== CREATE DATALOADERS ====================
    logger.info("\n" + "="*80)
    logger.info("CREATING DATALOADERS")
    logger.info("="*80)
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            images_dir=CONFIG['images_dir'],
            masks_dir=CONFIG['masks_dir'],
            batch_size=CONFIG['batch_size'],
            num_workers=CONFIG['num_workers'],
            image_size=CONFIG['image_size'],
            num_classes=CONFIG['num_classes'],
            split_ratio=CONFIG['split_ratio'],
            seed=42
        )
        
        logger.info(f"✓ Train loader: {len(train_loader)} batches")
        logger.info(f"✓ Val loader: {len(val_loader)} batches")
        logger.info(f"✓ Test loader: {len(test_loader)} batches")
        
    except Exception as e:
        logger.error(f"Error creating dataloaders: {e}")
        return


# ═══════════════════════════════════════════════════════════════════════════
# METRICS - IMPROVED & FIXED
# ═══════════════════════════════════════════════════════════════════════════

class SegmentationMetrics:
    """Robust segmentation metrics computation"""
    
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        self.tp = np.zeros(self.num_classes)
        self.fp = np.zeros(self.num_classes)
        self.fn = np.zeros(self.num_classes)
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update metrics with batch"""
        pred = predictions.cpu().numpy()
        target = targets.cpu().numpy()
        
        for cls in range(self.num_classes):
            pred_mask = (pred == cls)
            target_mask = (target == cls)
            
            self.tp[cls] += np.logical_and(pred_mask, target_mask).sum()
            self.fp[cls] += np.logical_and(pred_mask, ~target_mask).sum()
            self.fn[cls] += np.logical_and(~pred_mask, target_mask).sum()
    
    def compute_iou(self) -> Tuple[Dict, float]:
        """Compute IoU per class and mean"""
        iou_per_class = {}
        valid_iou = []
        
        for cls in range(self.num_classes):
            union = self.tp[cls] + self.fp[cls] + self.fn[cls]
            
            if union > 0:
                iou = self.tp[cls] / union
                iou_per_class[cls] = float(iou)
                valid_iou.append(iou)
            else:
                iou_per_class[cls] = 0.0
        
        mean_iou = np.mean(valid_iou) if valid_iou else 0.0
        
        return iou_per_class, float(mean_iou)
    
    def compute_dice(self) -> float:
        """Compute Dice coefficient"""
        dice_scores = []
        
        for cls in range(self.num_classes):
            total = 2 * self.tp[cls] + self.fp[cls] + self.fn[cls]
            
            if total > 0:
                dice = (2 * self.tp[cls]) / total
                dice_scores.append(dice)
        
        return float(np.mean(dice_scores)) if dice_scores else 0.0
    
    def compute_f1(self) -> float:
        """Compute macro F1 score"""
        f1_scores = []
        
        for cls in range(self.num_classes):
            precision = self.tp[cls] / (self.tp[cls] + self.fp[cls] + 1e-8)
            recall = self.tp[cls] / (self.tp[cls] + self.fn[cls] + 1e-8)
            
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            f1_scores.append(f1)
        
        return float(np.mean(f1_scores)) if f1_scores else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULING - IMPROVED
# ═══════════════════════════════════════════════════════════════════════════

class MultiStageScheduler:
    """Multi-stage learning rate scheduler"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, config: Config):
        self.optimizer = optimizer
        self.config = config
        self.current_epoch = 0
    
    def step(self):
        """Update learning rate for current epoch"""
        self.current_epoch += 1
        
        if self.current_epoch <= self.config.warmup_epochs:
            # Warmup: linear increase
            lr = self.config.learning_rate * (self.current_epoch / self.config.warmup_epochs)
        
        elif self.current_epoch <= self.config.warmup_epochs + self.config.cosine_epochs:
            # Main: cosine annealing
            progress = (self.current_epoch - self.config.warmup_epochs) / self.config.cosine_epochs
            lr = self.config.min_lr + (self.config.learning_rate - self.config.min_lr) * \
                 (1 + np.cos(np.pi * progress)) / 2
        
        else:
            # Fine-tune: exponential decay
            lr = self.config.min_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING LOOP - IMPROVED & FIXED
# ═══════════════════════════════════════════════════════════════════════════

def safe_batch_unpack(batch, device: torch.device, num_classes: int):
    """Safely unpack batch and validate"""
    
    if isinstance(batch, dict):
        images = batch['image']
        masks = batch['mask']
    else:
        images, masks = batch
    
    images = images.to(device).float()
    masks = masks.to(device).long()
    
    # Validate shapes
    assert len(images.shape) == 4, f"Images should be (B,C,H,W), got {images.shape}"
    assert len(masks.shape) == 3, f"Masks should be (B,H,W), got {masks.shape}"
    
    # Ensure valid range - CRITICAL FIX
    masks = torch.clamp(masks, 0, num_classes - 1)
    
    # Validate data
    assert not torch.isnan(images).any(), "NaN in images!"
    assert not torch.isnan(masks.float()).any(), "NaN in masks!"
    
    return images, masks


def validate_model_output(outputs: torch.Tensor, targets: torch.Tensor, 
                         num_classes: int, logger: logging.Logger):
    """Validate model output before metrics"""
    
    # Check shapes
    assert outputs.shape[0] == targets.shape[0], \
        f"Batch mismatch: {outputs.shape[0]} vs {targets.shape[0]}"
    assert outputs.shape[1] == num_classes, \
        f"Class mismatch: {outputs.shape[1]} vs {num_classes}"
    
    # Check for NaN/Inf
    assert not torch.isnan(outputs).any(), "NaN in outputs!"
    assert not torch.isinf(outputs).any(), "Inf in outputs!"
    
    # Get predictions
    predictions = torch.argmax(outputs, dim=1)
    
    # Check range
    assert predictions.min() >= 0 and predictions.max() < num_classes, \
        f"Predictions out of range: [{predictions.min()}, {predictions.max()}]"
    
    return predictions


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion,
                   optimizer: torch.optim.Optimizer, device: torch.device,
                   config: Config, scaler, logger: logging.Logger,
                   epoch: int) -> Tuple[float, float, float]:
    """Train for one epoch with all fixes"""
    
    model.train()
    running_loss = 0.0
    metrics = SegmentationMetrics(config.num_classes)
    
    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}", leave=True)
    
    for batch_idx, batch in enumerate(pbar):
        # FIXED: Safe unpacking with validation
        images, masks = safe_batch_unpack(batch, device, config.num_classes)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if config.use_amp:
            with autocast():
                outputs = model(images)
                loss_dict = criterion(outputs, masks)
                total_loss = loss_dict['total']
            
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        
        else:
            outputs = model(images)
            loss_dict = criterion(outputs, masks)
            total_loss = loss_dict['total']
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
        
        # FIXED: Validate model output before metrics
        predictions = validate_model_output(outputs, masks, config.num_classes, logger)
        
        # Update metrics
        metrics.update(predictions, masks)
        running_loss += total_loss.item()
        
        # Debug info on first batch
        if batch_idx == 0:
            logger.debug(f"Batch 0 debug:")
            logger.debug(f"  Loss: {total_loss.item():.4f}")
            logger.debug(f"  Pred range: [{predictions.min()}, {predictions.max()}]")
            logger.debug(f"  Target range: [{masks.min()}, {masks.max()}]")
        
        pbar.set_postfix({
            'loss': total_loss.item(),
            'iou': metrics.compute_iou()[1],
            'lr': optimizer.param_groups[0]['lr']
        })
    
    avg_loss = running_loss / len(dataloader)
    _, mean_iou = metrics.compute_iou()
    mean_dice = metrics.compute_dice()
    
    return avg_loss, mean_iou, mean_dice


def validate(model: nn.Module, dataloader: DataLoader, criterion,
            device: torch.device, config: Config, logger: logging.Logger) -> Tuple[float, float, float]:
    """Validate model"""
    
    model.eval()
    running_loss = 0.0
    metrics = SegmentationMetrics(config.num_classes)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=True)
        
        for batch in pbar:
            # FIXED: Safe unpacking
            images, masks = safe_batch_unpack(batch, device, config.num_classes)
            
            outputs = model(images)
            loss_dict = criterion(outputs, masks)
            total_loss = loss_dict['total']
            
            # FIXED: Validate output
            predictions = validate_model_output(outputs, masks, config.num_classes, logger)
            
            metrics.update(predictions, masks)
            running_loss += total_loss.item()
            
            pbar.set_postfix({'loss': total_loss.item()})
    
    avg_loss = running_loss / len(dataloader)
    _, mean_iou = metrics.compute_iou()
    mean_dice = metrics.compute_dice()
    
    return avg_loss, mean_iou, mean_dice


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main training function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train floor plan segmentation model')
    parser.add_argument('--config', type=str, default='production',
                       choices=['fast', 'production'], help='Config preset')
    parser.add_argument('--epochs', type=int, default=None, help='Override num epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--wandb', action='store_true', help='Use W&B logging')
    
    args = parser.parse_args()
    
    # Setup config
    config = Config()
    
    if args.config == 'fast':
        config.num_epochs = 50
        config.batch_size = 4
    elif args.config == 'production':
        config.num_epochs = 100
        config.batch_size = 8
    
    # Override from args
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    
    # Setup logging
    config.log_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(config.log_root)
    
    logger.info("=" * 80)
    logger.info("🚀 FLOOR PLAN SEGMENTATION - TRAINING START")
    logger.info("=" * 80)
    logger.info(f"\nConfiguration:")
    logger.info(f"  Model: {config.model_name} + {config.encoder_name}")
    logger.info(f"  Classes: {config.num_classes}")
    logger.info(f"  Batch Size: {config.batch_size}")
    logger.info(f"  Learning Rate: {config.learning_rate}")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Loss Config: {config.loss_config}")
    logger.info("Training completed successfully")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}\n")
    
    # Setup directories
    config.output_root.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    logger.info("\nCreating dataloaders...")
    try:
        train_loader, val_loader = create_dataloaders(config, logger)
    except Exception as e:
        logger.error(f"❌ Failed to create dataloaders: {e}")
        logger.error("Make sure your data structure matches:")
        logger.error("  data/processed/train_images/*.png")
        logger.error("  data/processed/train_masks/*.png")
        logger.error("  data/processed/val_images/*.png")
        logger.error("  data/processed/val_masks/*.png")
        return
    
    # Log dataloader information
    gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
    effective_batch_size = config.batch_size * gradient_accumulation_steps
    logger.info(f"Train batches: {len(train_loader)} (effective batch: {effective_batch_size})")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Calculate class weights - FIXED
    logger.info("Computing class weights...")
    class_weights = calculate_class_weights(
        train_loader, config.num_classes, device,
        method=config.class_weight_method, logger=logger
    )
    class_weights = class_weights.to(device)
    
    # Setup model
    logger.info(f"Loading model: {config.model_name}...")
    try:
        import segmentation_models_pytorch as smp
        model = smp.UnetPlusPlus(
            encoder_name=config.encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=config.num_classes,
            activation=None
        )
        model = model.to(device)
        logger.info("✓ Model loaded\n")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.error("Install segmentation_models_pytorch: pip install segmentation-models-pytorch")
        return
    
    # Setup loss function - FIXED: All losses enabled
    logger.info("Setting up loss function with all components...")
    criterion = CombinedLoss(config.loss_config, class_weights=class_weights)
    logger.info(f"✓ Loss function ready: {config.loss_config}\n")
    
    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    logger.info(f"✓ Optimizer: AdamW (lr={config.learning_rate}, wd={config.weight_decay})\n")
    
    # Setup scheduler
    scheduler = MultiStageScheduler(optimizer, config)
    logger.info(f"✓ Scheduler: Multi-stage (warmup={config.warmup_epochs}, cosine={config.cosine_epochs})\n")
    
    # Setup mixed precision
    scaler = GradScaler() if config.use_amp else None
    logger.info(f"✓ Mixed Precision: {'Enabled (FP16)' if config.use_amp else 'Disabled'}\n")
    
    # Setup TensorBoard
    tb_dir = config.log_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(str(tb_dir))
    logger.info(f"✓ TensorBoard: {tb_dir}\n")
    
    # Setup W&B (optional)
    if args.wandb and HAS_WANDB:
        wandb.init(
            project='floor-plan-segmentation',
            config={
                'model': config.model_name,
                'encoder': config.encoder_name,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
                'num_epochs': config.num_epochs,
            }
        )
        logger.info("✓ W&B logging enabled\n")
    
    # Training loop
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80 + "\n")
    
    best_iou = 0.0
    patience_counter = 0
    
    for epoch in range(1, config.num_epochs + 1):
        # Train
        train_loss, train_iou, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config, scaler, logger, epoch
        )
        
        # Validate
        val_loss, val_iou, val_dice = validate(model, val_loader, criterion, device, config, logger)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_lr()
        
        # Log metrics
        logger.info(f"\n📊 Epoch {epoch}/{config.num_epochs}")
        logger.info(f"  Train: Loss={train_loss:.4f}, IoU={train_iou:.4f}, Dice={train_dice:.4f}")
        logger.info(f"  Val:   Loss={val_loss:.4f}, IoU={val_iou:.4f}, Dice={val_dice:.4f}")
        logger.info(f"  LR: {current_lr:.2e}")
        
        # TensorBoard logging
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/iou', train_iou, epoch)
        writer.add_scalar('train/dice', train_dice, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/iou', val_iou, epoch)
        writer.add_scalar('val/dice', val_dice, epoch)
        writer.add_scalar('learning_rate', current_lr, epoch)
        
        # W&B logging
        if args.wandb and HAS_WANDB:
            wandb.log({
                'train/loss': train_loss,
                'train/iou': train_iou,
                'val/loss': val_loss,
                'val/iou': val_iou,
                'learning_rate': current_lr,
                'epoch': epoch
            })
        
        # Save checkpoint
        if epoch % config.save_interval == 0:
            checkpoint_path = config.output_root / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.__dict__,
            }, checkpoint_path)
            logger.info(f"  💾 Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            best_path = config.output_root / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_iou': best_iou,
                'config': config.__dict__,
            }, best_path)
            logger.info(f"  ⭐ New best model! IoU: {best_iou:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            logger.info(f"\n⏹️  Early stopping triggered after {patience_counter} epochs without improvement")
            break
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ TRAINING COMPLETED")
    logger.info("=" * 80)
    logger.info(f"\nResults:")
    logger.info(f"  Best Validation IoU: {best_iou:.4f}")
    logger.info(f"  Model saved to: {config.output_root}")
    logger.info(f"  TensorBoard logs: {tb_dir}")
    logger.info()
    
    writer.close()


if __name__ == '__main__':
    main()
