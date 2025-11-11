from PIL import Image
import numpy as np
import os

# Check the annotation image
anno_path = "data/cubicasa5k_converted/annotations/17.png"
if os.path.exists(anno_path):
    img = Image.open(anno_path)
    arr = np.array(img)
    
    print(f"Annotation Image: {anno_path}")
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")
    print(f"Min value: {arr.min()}")
    print(f"Max value: {arr.max()}")
    print(f"Unique values: {np.unique(arr)}")
    
    # Count pixels by value
    unique, counts = np.unique(arr, return_counts=True)
    print(f"\nPixel distribution:")
    for val, count in zip(unique, counts):
        pct = (count / arr.size) * 100
        print(f"  Value {val:3d}: {count:8d} pixels ({pct:5.2f}%)")
else:
    print(f"File not found: {anno_path}")
