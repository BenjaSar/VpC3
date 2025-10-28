import os
import shutil
from pathlib import Path

def convert_cubicasa_structure(source_dir, target_dir):
    """
    Convert CubiCasa5K to standard format
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    images_dir = target_path / "images"
    annotations_dir = target_path / "annotations"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images and annotations
    # Adjust paths based on CubiCasa5K structure
    for img_path in source_path.glob("**/*.png"):
        if "F1_original" in str(img_path):
            # Copy to images
            shutil.copy(img_path, images_dir / img_path.name)
        elif "model" in str(img_path):
            # Copy to annotations
            shutil.copy(img_path, annotations_dir / img_path.name)

convert_cubicasa_structure("cubicasa5k/cubicasa5k/high_quality", "image_converted")
