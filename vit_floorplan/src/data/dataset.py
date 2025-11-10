"""Dataset classes for floorplan segmentation."""
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Optional, Callable, Tuple


class FloorplanDataset(Dataset):
    """Custom dataset for floorplan segmentation."""
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Optional[Callable] = None,
        image_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg')
    ):
        """Initialize the dataset."""
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        
        self.images = []
        for ext in image_extensions:
            self.images.extend(sorted(self.image_dir.glob(f"*{ext}")))
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {image_dir}")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.images[idx]
        mask_path = self.mask_dir / img_path.name
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        image = np.array(image)
        mask = np.array(mask)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask
