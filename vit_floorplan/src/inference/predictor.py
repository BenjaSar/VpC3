"""Inference predictor class."""
import torch
import numpy as np
from PIL import Image
from typing import Union, Optional
from pathlib import Path


class Predictor:
    """Predictor for floorplan segmentation."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        transform: Optional[callable] = None
    ):
        self.model = model
        self.device = device
        self.transform = transform
        self.model.eval()
    
    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """Predict segmentation mask for an image."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        original_size = image.size
        image_np = np.array(image)
        
        if self.transform:
            transformed = self.transform(image=image_np)
            image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        
        output = self.model(image_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        pred_mask_pil = Image.fromarray(pred_mask.astype(np.uint8))
        pred_mask_pil = pred_mask_pil.resize(original_size, Image.NEAREST)
        
        return np.array(pred_mask_pil)
    
    def predict_batch(self, images: list) -> list:
        """Predict segmentation masks for a batch of images."""
        return [self.predict(img) for img in images]
