"""
Dataset Module for Floor Plan Images
Handles data loading, preprocessing, and augmentation
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)


class FloorPlanDataset(Dataset):
    """
    Floor Plan Dataset for semantic segmentation
    """
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        image_size: int = 512,
        augmentation: Optional[A.Compose] = None,
        split: str = "train",
        num_classes: int = 34
    ):
        """
        Initialize Floor Plan Dataset
        
        Args:
            images_dir: Path to images directory
            masks_dir: Path to masks directory
            image_size: Target image size
            augmentation: Albumentations augmentation pipeline
            split: Dataset split (train, val, test)
            num_classes: Number of classes
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.augmentation = augmentation
        self.split = split
        self.num_classes = num_classes
        
        # Get list of image files
        self.image_files = sorted([
            f for f in self.images_dir.glob('*.*')
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff']
        ])
        
        logger.info(f"Loaded {len(self.image_files)} images for {split} split")
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get single sample
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with image and mask tensors
        """
        try:
            # Load image
            img_path = self.image_files[idx]
            image = cv2.imread(str(img_path))
            
            if image is None:
                logger.warning(f"Failed to load image: {img_path}")
                return self.__getitem__((idx + 1) % len(self))
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load mask
            mask_path = self.masks_dir / (img_path.stem + '.png')
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                logger.warning(f"Failed to load mask: {mask_path}")
                return self.__getitem__((idx + 1) % len(self))
            
            # Apply augmentation
            if self.augmentation:
                augmented = self.augmentation(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
                # Albumentations ToTensorV2 already converts to tensor
                # So mask is already a tensor here
            else:
                # Default: resize and normalize
                image = cv2.resize(image, (self.image_size, self.image_size))
                mask = cv2.resize(mask, (self.image_size, self.image_size), 
                                 interpolation=cv2.INTER_NEAREST)
                
                # Normalize image
                image = image.astype(np.float32) / 255.0
                image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                image = torch.from_numpy(image).permute(2, 0, 1)
                
                # Convert mask to tensor (only if not using augmentation)
                mask = torch.from_numpy(mask).long()
            
            return {
                'image': image,
                'mask': mask,
                'filename': img_path.name
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))


def get_augmentation_pipeline(image_size: int = 512, split: str = "train") -> A.Compose:
    """
    Get albumentations augmentation pipeline
    Enhanced with aggressive augmentation to combat class imbalance
    
    Args:
        image_size: Target image size
        split: Dataset split (train, val, test)
        
    Returns:
        Albumentations Compose object
    """
    
    if split == "train":
        # PHASE 2: Aggressive augmentation to force model diversity
        augmentation = A.Compose([
            # Resize to target size
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            
            # GEOMETRIC AUGMENTATION: Force model to learn invariant features
            # Flips (floor plans are symmetric but not necessarily)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            
            # Rotations (small for floor plans - 0-20 degrees)
            A.Rotate(limit=20, p=0.6, border_mode=cv2.BORDER_REFLECT),
            
            # Shift, Scale, Rotate combined
            A.ShiftScaleRotate(
                shift_limit=0.1,      # Shift up to 10%
                scale_limit=0.15,     # Scale 0.85-1.15x
                rotate_limit=15,      # Rotate ±15 degrees
                p=0.5,
                border_mode=cv2.BORDER_REFLECT
            ),
            
            # Elastic deformation (morphs walls, spaces)
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=None,
                p=0.4
            ),
            
            # Grid distortion (warps floor plan layout slightly)
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                p=0.3,
                border_mode=cv2.BORDER_REFLECT
            ),
            
            # PHOTOMETRIC AUGMENTATION: Vary lighting conditions
            # Brightness and contrast (varying lighting)
            A.RandomBrightnessContrast(
                brightness_limit=0.3,    # ±30% brightness
                contrast_limit=0.3,      # ±30% contrast
                p=0.6
            ),
            
            # Gamma augmentation (simulate different camera exposures)
            A.RandomGamma(
                gamma_limit=(70, 130),
                p=0.3
            ),
            
            # Gaussian blur (simulate focus variations)
            A.GaussianBlur(
                blur_limit=3,
                p=0.3
            ),
            
            # Gaussian noise (simulate sensor noise)
            A.GaussNoise(
                var_limit=(10.0, 50.0),
                p=0.4
            ),
            
            # ISO noise (realistic camera noise)
            A.ISONoise(
                color_shift=(0.01, 0.05),
                intensity=(0.1, 0.5),
                p=0.2
            ),
            
            # Normalize to ImageNet statistics
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            
            # Convert to PyTorch tensors
            ToTensorV2()
        ], is_check_shapes=False)
    
    else:
        # Validation/Test: Minimal augmentation
        augmentation = A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], is_check_shapes=False)
    
    return augmentation


def create_dataloaders(
    images_dir: str,
    masks_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 512,
    num_classes: int = 34,
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        images_dir: Path to images directory
        masks_dir: Path to masks directory
        batch_size: Batch size
        num_workers: Number of workers
        image_size: Target image size
        num_classes: Number of classes
        split_ratio: Train/val/test split ratio
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Get all image files
    images_path = Path(images_dir)
    image_files = sorted([
        f for f in images_path.glob('*.*')
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff']
    ])
    
    # Split dataset
    np.random.seed(seed)
    indices = np.random.permutation(len(image_files))
    
    train_size = int(len(image_files) * split_ratio[0])
    val_size = int(len(image_files) * split_ratio[1])
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_files = [image_files[i] for i in train_indices]
    val_files = [image_files[i] for i in val_indices]
    test_files = [image_files[i] for i in test_indices]
    
    logger.info(f"Train samples: {len(train_files)}")
    logger.info(f"Val samples: {len(val_files)}")
    logger.info(f"Test samples: {len(test_files)}")
    
    # Create augmentation pipelines
    train_aug = get_augmentation_pipeline(image_size, "train")
    val_aug = get_augmentation_pipeline(image_size, "val")
    
    # Create datasets
    train_dataset = FloorPlanDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=image_size,
        augmentation=train_aug,
        split="train",
        num_classes=num_classes
    )
    
    val_dataset = FloorPlanDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=image_size,
        augmentation=val_aug,
        split="val",
        num_classes=num_classes
    )
    
    test_dataset = FloorPlanDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=image_size,
        augmentation=val_aug,
        split="test",
        num_classes=num_classes
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
