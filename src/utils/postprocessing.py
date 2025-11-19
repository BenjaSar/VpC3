#!/usr/bin/env python3
"""
Post-Processing Pipeline for Floor Plan Segmentation
Clean up predictions and remove artifacts
"""

import cv2
import numpy as np
from scipy.ndimage import label, binary_closing, binary_opening
from typing import Tuple


class SegmentationPostProcessor:
    """Post-process segmentation predictions"""
    
    def __init__(self, min_area=100, use_morphology=True):
        """
        Args:
            min_area: Minimum area (pixels) to keep a component
            use_morphology: Whether to apply morphological operations
        """
        self.min_area = min_area
        self.use_morphology = use_morphology
    
    def remove_small_regions(self, mask: np.ndarray, min_area: int = None) -> np.ndarray:
        """
        Remove isolated regions smaller than min_area
        
        Args:
            mask: Segmentation mask (H, W) with class indices
            min_area: Minimum area threshold (uses self.min_area if None)
        
        Returns:
            Cleaned mask
        """
        if min_area is None:
            min_area = self.min_area
        
        processed = mask.copy()
        
        # Process each class separately
        for class_id in range(1, 12):  # Skip background (class 0)
            binary_mask = (mask == class_id).astype(np.uint8)
            
            # Find connected components
            labeled, num_features = label(binary_mask)
            
            # Remove small components
            for region_id in range(1, num_features + 1):
                region_size = (labeled == region_id).sum()
                if region_size < min_area:
                    processed[labeled == region_id] = 0  # Set to background
        
        return processed
    
    def morphological_operations(self, mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply morphological operations to clean up mask
        
        Args:
            mask: Segmentation mask (H, W)
            kernel_size: Size of morphological kernel
        
        Returns:
            Processed mask
        """
        processed = mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        for class_id in range(1, 12):  # Skip background
            binary_mask = (mask == class_id).astype(np.uint8)
            
            # Morphological closing: fill small holes
            closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # Morphological opening: remove small noise
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Update mask
            processed[opened > 0] = class_id
            processed[(binary_mask > 0) & (opened == 0)] = 0  # Remove pixels not in opened
        
        return processed
    
    def smooth_boundaries(self, mask: np.ndarray, iterations: int = 1) -> np.ndarray:
        """
        Smooth class boundaries using morphological operations
        
        Args:
            mask: Segmentation mask (H, W)
            iterations: Number of smoothing iterations
        
        Returns:
            Smoothed mask
        """
        processed = mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        for _ in range(iterations):
            for class_id in range(1, 12):
                binary_mask = (processed == class_id).astype(np.uint8)
                
                # Dilate then erode to smooth
                dilated = cv2.dilate(binary_mask, kernel, iterations=1)
                smoothed = cv2.erode(dilated, kernel, iterations=1)
                
                processed[smoothed > 0] = class_id
        
        return processed
    
    def fill_gaps(self, mask: np.ndarray, gap_size: int = 3) -> np.ndarray:
        """
        Fill small gaps within regions
        
        Args:
            mask: Segmentation mask (H, W)
            gap_size: Maximum gap size to fill
        
        Returns:
            Filled mask
        """
        processed = mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_size, gap_size))
        
        for class_id in range(1, 12):
            binary_mask = (mask == class_id).astype(np.uint8)
            
            # Close (fill gaps)
            closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            processed[closed > 0] = class_id
        
        return processed
    
    def __call__(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply complete post-processing pipeline
        
        Args:
            mask: Segmentation mask (H, W) with class indices
        
        Returns:
            Post-processed mask
        """
        # Step 1: Fill small gaps
        mask = self.fill_gaps(mask, gap_size=3)
        
        # Step 2: Apply morphological operations
        if self.use_morphology:
            mask = self.morphological_operations(mask, kernel_size=5)
        
        # Step 3: Smooth boundaries
        mask = self.smooth_boundaries(mask, iterations=1)
        
        # Step 4: Remove small regions
        mask = self.remove_small_regions(mask, min_area=self.min_area)
        
        return mask


class CRFPostProcessor:
    """
    Conditional Random Field post-processing for refined boundaries
    (Optional: requires pydensecrf installation)
    """
    
    def __init__(self, num_classes=12, use_crf=False):
        """
        Args:
            num_classes: Number of segmentation classes
            use_crf: Whether to use CRF (requires pydensecrf)
        """
        self.num_classes = num_classes
        self.use_crf = use_crf
        
        if use_crf:
            try:
                import pydensecrf.densecrf as dcrf
                from pydensecrf.utils import unary_from_softmax
                self.dcrf = dcrf
                self.unary_from_softmax = unary_from_softmax
            except ImportError:
                self.use_crf = False
                print("Warning: pydensecrf not installed. CRF post-processing disabled.")
    
    def apply_crf(self, image: np.ndarray, softmax_output: np.ndarray) -> np.ndarray:
        """
        Apply CRF to refine segmentation boundaries
        
        Args:
            image: Input image (H, W, 3) in range [0, 255]
            softmax_output: Softmax probabilities (num_classes, H, W) in [0, 1]
        
        Returns:
            Refined mask (H, W)
        """
        if not self.use_crf:
            return softmax_output.argmax(axis=0)
        
        H, W = image.shape[:2]
        
        # Setup CRF
        crf = self.dcrf.DenseCRF2D(W, H, self.num_classes)
        
        # Add unary potential from softmax
        unary = self.unary_from_softmax(softmax_output)
        crf.setUnaryEnergy(unary)
        
        # Add pairwise energy (spatial smoothing)
        crf.addPairwiseGaussian(sxy=3, compat=3)
        
        # Add bilateral potential (texture-aware smoothing)
        crf.addPairwiseBilateral(
            sxy=10,
            srgb=13,
            rgbim=image,
            compat=10
        )
        
        # Inference
        Q = crf.inference(5)  # 5 iterations
        
        return np.argmax(Q, axis=0).astype(np.uint8)


def post_process_batch(
    masks: np.ndarray,
    min_area: int = 100,
    use_morphology: bool = True,
    use_crf: bool = False
) -> np.ndarray:
    """
    Post-process a batch of segmentation masks
    
    Args:
        masks: Batch of masks (B, H, W) with class indices
        min_area: Minimum area threshold
        use_morphology: Whether to apply morphological operations
        use_crf: Whether to apply CRF (if available)
    
    Returns:
        Post-processed masks (B, H, W)
    """
    processor = SegmentationPostProcessor(min_area=min_area, use_morphology=use_morphology)
    processed_masks = []
    
    for mask in masks:
        processed = processor(mask)
        processed_masks.append(processed)
    
    return np.stack(processed_masks, axis=0)


def post_process_single(
    mask: np.ndarray,
    min_area: int = 100,
    use_morphology: bool = True
) -> np.ndarray:
    """
    Post-process a single segmentation mask
    
    Args:
        mask: Segmentation mask (H, W) with class indices
        min_area: Minimum area threshold
        use_morphology: Whether to apply morphological operations
    
    Returns:
        Post-processed mask (H, W)
    """
    processor = SegmentationPostProcessor(min_area=min_area, use_morphology=use_morphology)
    return processor(mask)
