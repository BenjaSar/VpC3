"""Decoder architectures for segmentation."""
import torch
import torch.nn as nn


class SimpleDecoder(nn.Module):
    """Simple decoder with upsampling."""
    
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        x = self.decoder(x)
        x = nn.functional.interpolate(
            x, size=target_size, mode='bilinear', align_corners=False
        )
        return x
