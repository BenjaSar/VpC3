"""Post-processing utilities for predictions."""
import numpy as np
from scipy import ndimage


def remove_small_regions(mask: np.ndarray, min_size: int = 100) -> np.ndarray:
    """Remove small disconnected regions from mask."""
    labeled, num_features = ndimage.label(mask)
    
    for i in range(1, num_features + 1):
        region = labeled == i
        if np.sum(region) < min_size:
            mask[region] = 0
    
    return mask


def smooth_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply morphological smoothing to mask."""
    from scipy.ndimage import binary_closing, binary_opening
    
    smoothed = binary_closing(mask, structure=np.ones((kernel_size, kernel_size)))
    smoothed = binary_opening(smoothed, structure=np.ones((kernel_size, kernel_size)))
    
    return smoothed.astype(np.uint8)
