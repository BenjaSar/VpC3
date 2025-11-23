"""
Modelo Swin Transformer + Mask R-CNN
Arquitectura state-of-the-art para detección y segmentación
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class WindowAttention(nn.Module):
    """
    Window-based Multi-head Self Attention (W-MSA)
    Núcleo del Swin Transformer
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Initialize relative position bias
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B_, N, C = x.shape

        # QKV
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SwinTransformerBlock(nn.Module):
    """
    Bloque básico del Swin Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # Layer Norm
        self.norm1 = nn.LayerNorm(dim)

        # Window Attention
        self.attn = WindowAttention(
            dim,
            window_size=(window_size, window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # MLP
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor):
        # Window Attention
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x

        # MLP
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x

        return x


class SwinTransformerBackbone(nn.Module):
    """
    Swin Transformer Backbone
    Extrae features multi-escala para detección
    """

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Absolute position embedding
        num_patches = (img_size // patch_size) ** 2
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)

        # Layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = self._make_layer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                downsample=(i_layer < self.num_layers - 1),
            )
            self.layers.append(layer)

        # Layer norm for each output
        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.num_features[i]) for i in range(self.num_layers)]
        )

    def _make_layer(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        downsample: bool,
    ):
        blocks = []
        for i in range(depth):
            blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                )
            )

        # Downsampling layer
        if downsample:
            blocks.append(
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * 2),
                    nn.Unflatten(1, (2, -1)),  # For spatial downsampling
                )
            )

        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            List of multi-scale features
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/4, W/4)

        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Add position embedding
        x = x + self.absolute_pos_embed

        # Extract multi-scale features
        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Reshape to spatial
            if i < self.num_layers - 1:
                H, W = H // 2, W // 2

            # Apply norm and reshape
            x_out = self.norms[i](x)
            x_out = x_out.transpose(1, 2).reshape(B, -1, H, W)
            features.append(x_out)

        return features


class FPN(nn.Module):
    """
    Feature Pyramid Network
    Combina features multi-escala
    """

    def __init__(
        self, in_channels_list: List[int], out_channels: int, num_outs: int = 5
    ):
        super().__init__()
        self.num_outs = num_outs

        # Lateral connections
        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, 1)
                for in_channels in in_channels_list
            ]
        )

        # Output convolutions
        self.fpn_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
                for _ in range(len(in_channels_list))
            ]
        )

        # Extra layers for more pyramid levels
        if num_outs > len(in_channels_list):
            self.extra_convs = nn.ModuleList(
                [
                    nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
                    for _ in range(num_outs - len(in_channels_list))
                ]
            )

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            inputs: Multi-scale features from backbone

        Returns:
            FPN feature pyramid
        """
        # Build lateral connections
        laterals = [
            lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Build top-down path
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )

        # Build outputs
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]

        # Add extra pyramid levels
        if hasattr(self, "extra_convs"):
            for i, extra_conv in enumerate(self.extra_convs):
                if i == 0:
                    outs.append(extra_conv(outs[-1]))
                else:
                    outs.append(extra_conv(F.relu(outs[-1])))

        return outs


class RoomDetectionHead(nn.Module):
    """
    Head para detección de habitaciones
    """

    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)

        # Classification
        self.cls_head = nn.Conv2d(hidden_dim, num_classes, 1)

        # Regression (bbox)
        self.reg_head = nn.Conv2d(hidden_dim, 4, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        cls_score = self.cls_head(x)
        bbox_pred = self.reg_head(x)

        return cls_score, bbox_pred


class MaskHead(nn.Module):
    """
    Head para segmentación de máscaras
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_convs: int = 4,
    ):
        super().__init__()

        # Convolutional layers
        convs = []
        for i in range(num_convs):
            in_c = in_channels if i == 0 else hidden_dim
            convs.append(nn.Conv2d(in_c, hidden_dim, 3, padding=1))
            convs.append(nn.ReLU(inplace=True))

        self.convs = nn.Sequential(*convs)

        # Upsample
        self.upsample = nn.ConvTranspose2d(hidden_dim, hidden_dim, 2, stride=2)

        # Mask prediction
        self.mask_head = nn.Conv2d(hidden_dim, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        x = F.relu(self.upsample(x))
        mask = self.mask_head(x)
        return mask


class SwinMaskRCNN(nn.Module):
    """
    Modelo completo: Swin Transformer + Mask R-CNN
    """

    def __init__(
        self,
        num_classes: int = 15,
        img_size: int = 512,
        backbone_config: Optional[Dict] = None,
        pretrained: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes

        # Backbone: Swin Transformer
        backbone_config = backbone_config or {}
        self.backbone = SwinTransformerBackbone(img_size=img_size, **backbone_config)

        # Neck: FPN
        self.neck = FPN(
            in_channels_list=self.backbone.num_features, out_channels=256, num_outs=5
        )

        # Detection head
        self.detection_head = RoomDetectionHead(
            in_channels=256, num_classes=num_classes, hidden_dim=256
        )

        # Mask head
        self.mask_head = MaskHead(
            in_channels=256, num_classes=num_classes, hidden_dim=256, num_convs=4
        )

    def forward(
        self, images: torch.Tensor, targets: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: Input images (B, C, H, W)
            targets: Ground truth (for training)

        Returns:
            Dictionary with predictions
        """
        # Extract features
        backbone_features = self.backbone(images)

        # FPN
        fpn_features = self.neck(backbone_features)

        # Detection
        cls_scores = []
        bbox_preds = []
        for feat in fpn_features:
            cls, bbox = self.detection_head(feat)
            cls_scores.append(cls)
            bbox_preds.append(bbox)

        # Masks (usar el feature más fino)
        masks = self.mask_head(fpn_features[0])

        outputs = {"cls_scores": cls_scores, "bbox_preds": bbox_preds, "masks": masks}

        return outputs


if __name__ == "__main__":
    # Test del modelo
    print("🏗️  Testing Swin Mask R-CNN...")

    model = SwinMaskRCNN(num_classes=15, img_size=512)

    # Input de prueba
    x = torch.randn(2, 3, 512, 512)

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    outputs = model(x)

    print("\nOutput shapes:")
    print(f"  - cls_scores: {len(outputs['cls_scores'])} scales")
    print(f"  - bbox_preds: {len(outputs['bbox_preds'])} scales")
    print(f"  - masks: {outputs['masks'].shape}")

    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\n✅ Model test passed!")
