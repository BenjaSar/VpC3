#!/usr/bin/env python3
"""
Runnable Script: Data Preprocessing
Uses src/preprocessing.py to validate and preprocess your dataset
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import ImagePreprocessor, DataValidator  
from src.utils.logging_config import setup_logging
import cv2
# import numpy as np
from tqdm import tqdm

logger = setup_logging()


def main():
    """
    Main preprocessing pipeline
    """
    
    # ==================== CONFIGURATION ====================
    INPUT_DIR = Path("data/cubicasa5k_converted")
    OUTPUT_DIR = Path("data/processed")
    TARGET_SIZE = 512
    REMOVE_INVALID = False  # Set to True to remove invalid files
    
    # ==================== STEP 1: VALIDATION ====================
    logger.info("="*80)
    logger.info("STEP 1: VALIDATING DATASET")
    logger.info("="*80)
    
    images_dir = INPUT_DIR / "images"
    masks_dir = INPUT_DIR / "annotations"
    
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        logger.info("Have you converted the CubiCasa5K dataset?")
        logger.info("Run: python data/svg_to_png_converter.py")
        return
    
    if not masks_dir.exists():
        logger.error(f"Annotations directory not found: {masks_dir}")
        return
    
    # Check image-annotation matching
    logger.info("Checking image-annotation matching...")
    all_match, missing = DataValidator.check_image_annotation_match(images_dir, masks_dir)
    
    if all_match:
        logger.info("✓ All images have matching annotations")
    else:
        logger.warning(f"✗ {len(missing)} images missing annotations")
        logger.warning(f"First 5: {missing[:5]}")
    
    # Validate dataset
    logger.info("\nValidating dataset integrity...")
    valid_count, invalid_count, invalid_files = DataValidator.validate_dataset(
        images_dir, masks_dir, remove_invalid=False
    )
    
    logger.info(f"Valid samples: {valid_count}")
    logger.info(f"Invalid samples: {invalid_count}")
    
    if invalid_count > 0:
        logger.warning(f"Found {invalid_count} invalid files (will be skipped)")
        if not REMOVE_INVALID:
            logger.info("Continuing with valid files only...")
            logger.info("To remove invalid files, set REMOVE_INVALID=True in the script")
    
    # ==================== STEP 2: PREPROCESSING ====================
    logger.info("\n" + "="*80)
    logger.info("STEP 2: PREPROCESSING IMAGES")
    logger.info("="*80)
    
    # Create output directories
    output_images = OUTPUT_DIR / "images"
    output_masks = OUTPUT_DIR / "annotations"
    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)
    
    # Get image files
    image_files = sorted(list(images_dir.glob("*.png")))
    logger.info(f"Processing {len(image_files)} images to {TARGET_SIZE}x{TARGET_SIZE}...")
    
    preprocessor = ImagePreprocessor()
    processed_count = 0
    
    for img_path in tqdm(image_files, desc="Preprocessing"):
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Skipping {img_path.name}")
                continue
            
            # Load mask
            mask_path = masks_dir / img_path.name
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"Skipping {img_path.name} - no mask")
                continue
            
            # Resize (use NEAREST for masks to preserve class IDs)
            img_resized = preprocessor.resize_image(img, TARGET_SIZE)
            mask_resized = preprocessor.resize_image(
                mask, TARGET_SIZE, 
                interpolation=cv2.INTER_NEAREST
            )
            
            # Save preprocessed data
            output_img = output_images / img_path.name
            output_mask = output_masks / img_path.name
            
            cv2.imwrite(str(output_img), img_resized)
            cv2.imwrite(str(output_mask), mask_resized)
            
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {e}")
    
    # ==================== STEP 3: FINAL VALIDATION ====================
    logger.info("\n" + "="*80)
    logger.info("STEP 3: VALIDATING PREPROCESSED DATA")
    logger.info("="*80)
    
    valid, invalid, _ = DataValidator.validate_dataset(
        output_images, output_masks
    )
    
    logger.info(f"Processed dataset: {valid} valid, {invalid} invalid")
    
    # ==================== SUMMARY ====================
    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING COMPLETE!")
    logger.info("="*80)
    logger.info(f"✓ Processed: {processed_count}/{len(image_files)} images")
    logger.info(f"✓ Output directory: {OUTPUT_DIR}")
    logger.info(f"✓ Image size: {TARGET_SIZE}x{TARGET_SIZE}")
    logger.info("\nNext steps:")
    logger.info("1. Run EDA: python src/eda/eda_analysis.py --dataset_path data/processed")
    logger.info("2. Create dataloaders: python run_dataset.py")
    logger.info("3. Start training: python train.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
    except Exception as e:
        logger.error(f"\n\nFatal error: {e}", exc_info=True)
