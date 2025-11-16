import numpy as np
from pathlib import Path
from PIL import Image

masks_dir = Path("data/processed/annotations")
mask_files = list(masks_dir.glob("*.png"))

print(f"Found {len(mask_files)} mask files\n")

for mask_file in mask_files[:10]:
    try:
        mask = np.array(Image.open(mask_file))
        unique_vals = np.unique(mask)
        print(f"{mask_file.name}:")
        print(f"  Shape: {mask.shape}")
        print(f"  Min: {mask.min()}, Max: {mask.max()}")
        print(f"  Unique values: {unique_vals}")
        if mask.max() > 11 or mask.min() < 0:
            print(f"  ⚠️  WARNING: Values out of range [0, 11]!")
        print()
    except Exception as e:
        print(f"Error reading {mask_file.name}: {e}\n")

print("Verification complete!")
