import torch
import numpy as np
from pathlib import Path
from PIL import Image

def calculate_class_weights(data_dir):
    """Calculate inverse frequency class weights."""
    counts = np.zeros(256)
    mask_files = list(Path(data_dir).glob('*.png'))
    
    for mask_file in mask_files:
        mask = np.array(Image.open(mask_file))
        unique, count = np.unique(mask, return_counts=True)
        counts[unique] += count
    
    # Convert counts to weights
    total = np.sum(counts)
    weights = np.zeros_like(counts, dtype=np.float32)
    for i in range(len(counts)):
        if counts[i] > 0:
            weights[i] = total / (counts[i] * len(counts))
    
    # Normalize weights
    weights = weights / np.sum(weights)
    return torch.FloatTensor(weights)

if __name__ == '__main__':
    weights = calculate_class_weights('data/processed/annotations')
    torch.save(weights, 'class_weights.pt')
    print("Class weights saved to class_weights.pt")