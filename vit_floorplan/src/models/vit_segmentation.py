"""Vision Transformer for semantic segmentation."""
import torch
import torch.nn as nn
from transformers import ViTModel


class ViTSegmentation(nn.Module):
    """Vision Transformer model for semantic segmentation."""
    
    def __init__(
        self,
        num_classes: int = 12,
        pretrained: bool = True,
        model_name: str = 'google/vit-base-patch16-224'
    ):
        super().__init__()
        self.num_classes = num_classes
        
        if pretrained:
            self.vit = ViTModel.from_pretrained(model_name)
        else:
            from transformers import ViTConfig
            config = ViTConfig.from_pretrained(model_name)
            self.vit = ViTModel(config)
        
        hidden_size = self.vit.config.hidden_size
        
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_size, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        outputs = self.vit(x)
        features = outputs.last_hidden_state[:, 1:, :]
        
        num_patches = features.shape[1]
        patch_h = patch_w = int(num_patches ** 0.5)
        
        features = features.reshape(B, patch_h, patch_w, -1)
        features = features.permute(0, 3, 1, 2)
        
        logits = self.decoder(features)
        logits = nn.functional.interpolate(
            logits, size=(H, W), mode='bilinear', align_corners=False
        )
        
        return logits
