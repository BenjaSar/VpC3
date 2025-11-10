"""
Data Preprocessing Module
Handles data normalization, resizing, and validation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Image preprocessing utilities
    """
    
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    
    @staticmethod
    def resize_image(
        image: np.ndarray,
        size: int,
        interpolation: int = cv2.INTER_LINEAR
    ) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image
            size: Target size (square)
            interpolation: Interpolation method
            
        Returns:
            Resized image
        """
        return cv2.resize(image, (size, size), interpolation=interpolation)
    
    @staticmethod
    def normalize_image(
        image: np.ndarray,
        mean: np.ndarray = None,
        std: np.ndarray = None
    ) -> np.ndarray:
        """
        Normalize image using ImageNet statistics
        
        Args:
            image: Input image (0-255 or 0-1)
            mean: Normalization mean
            std: Normalization std
            
        Returns:
            Normalized image
        """
        if mean is None:
            mean = ImagePreprocessor.IMAGENET_MEAN
        if std is None:
            std = ImagePreprocessor.IMAGENET_STD
        
        # Convert to float if needed
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        
        # Normalize
        image = (image - mean) / std
        
        return image
    
    @staticmethod
    def validate_image(image_path: Path) -> bool:
        """
        Validate image file
        
        Args:
            image_path: Path to image
            
        Returns:
            True if valid, False otherwise
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning(f"Failed to load image: {image_path}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error validating image {image_path}: {str(e)}")
            return False
    
    @staticmethod
    def validate_mask(mask_path: Path) -> bool:
        """
        Validate mask file
        
        Args:
            mask_path: Path to mask
            
        Returns:
            True if valid, False otherwise
        """
        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"Failed to load mask: {mask_path}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error validating mask {mask_path}: {str(e)}")
            return False


class DataValidator:
    """
    Data validation utilities
    """
    
    @staticmethod
    def check_image_annotation_match(
        images_dir: Path,
        masks_dir: Path
    ) -> Tuple[bool, List[str]]:
        """
        Check if all images have corresponding annotations
        
        Args:
            images_dir: Images directory
            masks_dir: Masks directory
            
        Returns:
            Tuple of (all_match, missing_files)
        """
        missing = []
        image_files = list(images_dir.glob('*.*'))
        
        for img_file in image_files:
            mask_file = masks_dir / (img_file.stem + '.png')
            if not mask_file.exists():
                missing.append(str(img_file))
        
        return len(missing) == 0, missing
    
    @staticmethod
    def validate_dataset(
        images_dir: Path,
        masks_dir: Path,
        remove_invalid: bool = False
    ) -> Tuple[int, int, List[str]]:
        """
        Validate entire dataset
        
        Args:
            images_dir: Images directory
            masks_dir: Masks directory
            remove_invalid: Whether to remove invalid files
            
        Returns:
            Tuple of (valid_count, invalid_count, invalid_files)
        """
        valid_count = 0
        invalid_count = 0
        invalid_files = []
        
        image_files = list(images_dir.glob('*.*'))
        
        for img_file in image_files:
            mask_file = masks_dir / (img_file.stem + '.png')
            
            # Check if mask exists
            if not mask_file.exists():
                invalid_files.append(str(img_file))
                invalid_count += 1
                if remove_invalid:
                    img_file.unlink()
                continue
            
            # Validate image
            if not ImagePreprocessor.validate_image(img_file):
                invalid_files.append(str(img_file))
                invalid_count += 1
                if remove_invalid:
                    img_file.unlink()
                    mask_file.unlink()
                continue
            
            # Validate mask
            if not ImagePreprocessor.validate_mask(mask_file):
                invalid_files.append(str(mask_file))
                invalid_count += 1
                if remove_invalid:
                    img_file.unlink()
                    mask_file.unlink()
                continue
            
            valid_count += 1
        
        logger.info(f"Dataset validation: {valid_count} valid, {invalid_count} invalid")
        
        return valid_count, invalid_count, invalid_files