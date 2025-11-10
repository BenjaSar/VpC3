#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Creating Complete Cookiecutter Template${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Set template directory
TEMPLATE_DIR=~/cookiecutter-floorplan-vit
mkdir -p "$TEMPLATE_DIR"
cd "$TEMPLATE_DIR"

echo -e "${GREEN}✓ Created template directory: $TEMPLATE_DIR${NC}"

# ============================================
# 1. CREATE cookiecutter.json
# ============================================
echo -e "${YELLOW}Creating cookiecutter.json...${NC}"

cat > cookiecutter.json << 'EOF'
{
    "project_name": "vit-floorplan",
    "project_slug": "{{ cookiecutter.project_name.lower().replace(' ', '_').replace('-', '_') }}",
    "author_name": "FS, Alejandro Lloveras, Jorge Cuenca",
    "author_email": "fs@example.com",
    "github_username": "BenjaSar",
    "description": "Vision Transformer for floorplan segmentation and classification",
    "version": "0.1.0",
    "python_version": ["3.10", "3.11", "3.12.3"],
    "cuda_version": "11.8",
    "use_docker": ["yes", "no"],
    "use_mlflow": ["yes", "no"],
    "use_github_actions": ["yes", "no"],
    "license": ["MIT", "Apache-2.0", "GPL-3.0", "BSD-3-Clause"]
}

EOF

echo -e "${GREEN}✓ cookiecutter.json created${NC}"

# ============================================
# 2. CREATE TEMPLATE DIRECTORY STRUCTURE
# ============================================
echo -e "${YELLOW}Creating directory structure...${NC}"

PROJECT_DIR="{{cookiecutter.project_slug}}"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create all directories
mkdir -p .github/workflows
mkdir -p src/{config,data,models,training,inference,utils,eda}
mkdir -p notebooks
mkdir -p tests
mkdir -p configs
mkdir -p scripts
mkdir -p docker
mkdir -p requirements
mkdir -p mlruns

echo -e "${GREEN}✓ Directory structure created${NC}"

# ============================================
# 3. GITHUB WORKFLOWS
# ============================================
echo -e "${YELLOW}Creating GitHub workflows...${NC}"

cat > .github/workflows/ci.yml << 'EOF'
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["{{cookiecutter.python_version}}"]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: {% raw %}${{ matrix.python-version }}{% endraw %}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements/dev.txt
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=src
    
    - name: Lint with flake8
      run: |
        flake8 src/ --max-line-length=120
    
    - name: Format check with black
      run: |
        black --check src/
EOF

cat > .github/workflows/deploy.yml << 'EOF'
name: Deploy

on:
  push:
    tags:
      - 'v*'

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "{{cookiecutter.python_version}}"
    
    - name: Build Docker image
      run: |
        docker build -f docker/Dockerfile -t {{cookiecutter.project_slug}}:latest .
    
    - name: Deploy
      run: |
        echo "Add your deployment steps here"
EOF

# ============================================
# 4. SOURCE CODE - CONFIG
# ============================================
echo -e "${YELLOW}Creating config files...${NC}"

cat > src/__init__.py << 'EOF'
"""{{cookiecutter.description}}"""

__version__ = "{{cookiecutter.version}}"
__author__ = "{{cookiecutter.author_name}}"
__email__ = "{{cookiecutter.author_email}}"
EOF

touch src/config/__init__.py

cat > src/config/config.yaml << 'EOF'
# Model Configuration
model:
  name: "vit_base_patch16"
  pretrained: true
  num_classes: 12
  image_size: 512
  patch_size: 16

# Training Configuration
training:
  batch_size: 16
  epochs: 100
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  early_stopping_patience: 10
  gradient_clip: 1.0
  mixed_precision: true

# Data Configuration
data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  num_workers: 4
  pin_memory: true
  
# Augmentation
augmentation:
  horizontal_flip: 0.5
  vertical_flip: 0.5
  rotation_limit: 15
  brightness_contrast: 0.3

# Paths
paths:
  data_dir: "data/raw"
  processed_dir: "data/processed"
  models_dir: "models"
  logs_dir: "logs"
  checkpoints_dir: "checkpoints"
EOF

cat > src/config/paths.py << 'EOF'
"""Path configuration for the project."""
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Log directories
LOGS_DIR = PROJECT_ROOT / "logs"
TENSORBOARD_DIR = LOGS_DIR / "tensorboard"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"

# Config directory
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Create directories if they don't exist
for directory in [
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    MODELS_DIR, CHECKPOINTS_DIR,
    LOGS_DIR, TENSORBOARD_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
EOF

# ============================================
# 5. SOURCE CODE - DATA
# ============================================
echo -e "${YELLOW}Creating data module...${NC}"

touch src/data/__init__.py

cat > src/data/dataset.py << 'EOF'
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
EOF

cat > src/data/preprocessing.py << 'EOF'
"""Data preprocessing utilities."""
import numpy as np
from PIL import Image
from typing import Tuple, Union


def resize_image(
    image: Union[Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (512, 512)
) -> Union[Image.Image, np.ndarray]:
    """Resize image to target size."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return image.resize(target_size, Image.BILINEAR)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range."""
    return image.astype(np.float32) / 255.0


def preprocess_mask(mask: np.ndarray, num_classes: int = 12) -> np.ndarray:
    """Preprocess segmentation mask."""
    mask = np.clip(mask, 0, num_classes - 1)
    return mask.astype(np.int64)
EOF

cat > src/data/augmentation.py << 'EOF'
"""Data augmentation strategies."""
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_augmentation(image_size: int = 512) -> A.Compose:
    """Get training augmentation pipeline."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5, border_mode=0),
        A.RandomBrightnessContrast(p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_validation_augmentation(image_size: int = 512) -> A.Compose:
    """Get validation augmentation pipeline."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
EOF

# ============================================
# 6. SOURCE CODE - MODELS
# ============================================
echo -e "${YELLOW}Creating models module...${NC}"

touch src/models/__init__.py

cat > src/models/vit_segmentation.py << 'EOF'
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
EOF

cat > src/models/decoder.py << 'EOF'
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
EOF

cat > src/models/loss_functions.py << 'EOF'
"""Custom loss functions for segmentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1])
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined Cross Entropy and Dice loss."""
    
    def __init__(self, weight_ce: float = 0.5, weight_dice: float = 0.5):
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.weight_ce * self.ce(pred, target) + self.weight_dice * self.dice(pred, target)
EOF

# ============================================
# 7. SOURCE CODE - TRAINING
# ============================================
echo -e "${YELLOW}Creating training module...${NC}"

touch src/training/__init__.py

cat > src/training/trainer.py << 'EOF'
"""Training loop implementation."""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional
import logging


class Trainer:
    """Trainer class for model training."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Dict
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        self.logger = logging.getLogger(__name__)
        
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1} [Train]')
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc=f'Epoch {self.current_epoch+1} [Val]'):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def fit(self, epochs: int):
        """Train the model."""
        self.logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
                self.logger.info(f"✓ Saved best model (val_loss: {val_loss:.4f})")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
EOF

cat > src/training/callbacks.py << 'EOF'
"""Training callbacks."""
import logging


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            self.logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False
EOF

cat > src/training/metrics.py << 'EOF'
"""Evaluation metrics for segmentation."""
import torch
import numpy as np
from typing import Dict


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """Calculate IoU for each class."""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())
    
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0
    
    return {'iou_per_class': ious, 'mean_iou': mean_iou}


def calculate_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate pixel accuracy."""
    correct = (pred == target).sum()
    total = target.numel()
    return (correct / total).item()
EOF

# ============================================
# 8. SOURCE CODE - INFERENCE
# ============================================
echo -e "${YELLOW}Creating inference module...${NC}"

touch src/inference/__init__.py

cat > src/inference/predictor.py << 'EOF'
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
EOF

cat > src/inference/postprocessing.py << 'EOF'
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
EOF

# ============================================
# 9. SOURCE CODE - UTILS
# ============================================
echo -e "${YELLOW}Creating utils module...${NC}"

touch src/utils/__init__.py

cat > src/utils/logging.py << 'EOF'
"""Logging utilities."""
import logging
from pathlib import Path


def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """Set up a logger with file and console handlers."""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    
    return logger
EOF

cat > src/utils/visualization.py << 'EOF'
"""Visualization utilities."""
import matplotlib.pyplot as plt
import numpy as np


def visualize_prediction(image, mask, prediction, class_names=None):
    """Visualize image, ground truth, and prediction."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='tab20')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(prediction, cmap='tab20')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def plot_training_history(history):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training History')
    ax1.legend()
    
    if 'train_iou' in history:
        ax2.plot(history['train_iou'], label='Train IoU')
        ax2.plot(history['val_iou'], label='Val IoU')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('IoU')
        ax2.set_title('IoU History')
        ax2.legend()
    
    plt.tight_layout()
    return fig
EOF

# ============================================
# 10. SOURCE CODE - EDA
# ============================================
echo -e "${YELLOW}Creating EDA module...${NC}"

touch src/eda/__init__.py

cat > src/eda/dataset_analysis.py << 'EOF'
"""Dataset analysis utilities."""
import numpy as np
from pathlib import Path
from collections import Counter


def analyze_dataset_structure(data_dir):
    """Analyze the structure of the dataset."""
    data_path = Path(data_dir)
    
    info = {
        'total_images': 0,
        'total_masks': 0,
        'image_formats': Counter(),
        'mask_formats': Counter()
    }
    
    if (data_path / 'images').exists():
        images = list((data_path / 'images').glob('*'))
        info['total_images'] = len(images)
        info['image_formats'] = Counter([img.suffix for img in images])
    
    if (data_path / 'masks').exists():
        masks = list((data_path / 'masks').glob('*'))
        info['total_masks'] = len(masks)
        info['mask_formats'] = Counter([mask.suffix for mask in masks])
    
    return info
EOF

cat > src/eda/class_distribution.py << 'EOF'
"""Analyze class distribution in masks."""
import numpy as np
from PIL import Image
from collections import Counter
from pathlib import Path


def analyze_class_distribution(mask_dir, num_classes=12):
    """Analyze class distribution in segmentation masks."""
    mask_path = Path(mask_dir)
    mask_files = list(mask_path.glob('*.png'))
    
    class_counts = Counter()
    total_pixels = 0
    
    for mask_file in mask_files:
        mask = np.array(Image.open(mask_file))
        unique, counts = np.unique(mask, return_counts=True)
        
        for cls, count in zip(unique, counts):
            class_counts[cls] += count
            total_pixels += count
    
    class_percentages = {cls: (count / total_pixels) * 100 
                        for cls, count in class_counts.items()}
    
    return {
        'class_counts': dict(class_counts),
        'class_percentages': class_percentages,
        'total_pixels': total_pixels
    }


def calculate_class_weights(class_counts, method='inverse'):
    """Calculate class weights for imbalanced data."""
    total = sum(class_counts.values())
    
    if method == 'inverse':
        weights = {cls: total / count for cls, count in class_counts.items()}
    else:
        weights = {cls: 1.0 for cls in class_counts.keys()}
    
    max_weight = max(weights.values())
    weights = {cls: w / max_weight for cls, w in weights.items()}
    
    return weights
EOF

cat > src/eda/visualizations.py << 'EOF'
"""EDA visualization utilities."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_class_distribution(class_counts, title="Class Distribution"):
    """Plot class distribution bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    ax.bar(classes, counts)
    ax.set_xlabel('Class')
    ax.set_ylabel('Pixel Count')
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


def plot_image_size_distribution(sizes):
    """Plot distribution of image sizes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    size_counts = {}
    for size in sizes:
        size_str = f"{size[0]}x{size[1]}"
        size_counts[size_str] = size_counts.get(size_str, 0) + 1
    
    ax.bar(size_counts.keys(), size_counts.values())
    ax.set_xlabel('Image Size')
    ax.set_ylabel('Count')
    ax.set_title('Image Size Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig
EOF

# ============================================
# 11. SCRIPTS
# ============================================
echo -e "${YELLOW}Creating scripts...${NC}"

cat > scripts/eda_analysis.py << 'EOF'
"""⭐ EDA Analysis Script - First Step."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from eda.dataset_analysis import analyze_dataset_structure
from eda.class_distribution import analyze_class_distribution, calculate_class_weights
import json


def main():
    """Run comprehensive EDA analysis."""
    print("=" * 60)
    print("FLOORPLAN DATASET - EDA ANALYSIS")
    print("=" * 60)
    
    data_dir = Path("data/raw")
    
    print("\n1. Dataset Structure Analysis")
    print("-" * 60)
    structure_info = analyze_dataset_structure(data_dir)
    print(json.dumps(structure_info, indent=2))
    
    print("\n2. Class Distribution Analysis")
    print("-" * 60)
    mask_dir = data_dir / "masks"
    if mask_dir.exists():
        class_info = analyze_class_distribution(mask_dir)
        print(json.dumps(class_info, indent=2))
        
        weights = calculate_class_weights(class_info['class_counts'])
        print("\n3. Calculated Class Weights")
        print("-" * 60)
        print(json.dumps(weights, indent=2))
    else:
        print("Mask directory not found. Please download dataset first.")
    
    print("\n" + "=" * 60)
    print("EDA Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
EOF

cat > scripts/download_dataset.py << 'EOF'
"""Script to download the dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config.paths import RAW_DATA_DIR


def main():
    """Download dataset."""
    print("Dataset download script")
    print(f"Target directory: {RAW_DATA_DIR}")
    print("\n⚠️  Please implement dataset download logic here.")
    print("You can use:")
    print("  - kaggle API")
    print("  - wget/curl")
    print("  - Custom download script")


if __name__ == "__main__":
    main()
EOF

cat > scripts/train.py << 'EOF'
"""Training script."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import yaml
from models.vit_segmentation import ViTSegmentation
from models.loss_functions import CombinedLoss
from training.trainer import Trainer


def main():
    """Main training function."""
    with open('configs/train_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ViTSegmentation(num_classes=config['model']['num_classes'])
    model = model.to(device)
    
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    print("Training script ready. Add data loaders to start training.")
    print("See notebooks/02_model_training.ipynb for full example.")


if __name__ == "__main__":
    main()
EOF

cat > scripts/evaluate.py << 'EOF'
"""Evaluation script."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def main():
    """Evaluate model on test set."""
    print("Evaluation script")
    print("Implement model evaluation logic here")


if __name__ == "__main__":
    main()
EOF

cat > scripts/inference.py << 'EOF'
"""Inference script."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def main():
    """Run inference on new images."""
    print("Inference script")
    print("Implement inference logic here")


if __name__ == "__main__":
    main()
EOF

cat > scripts/export_model.py << 'EOF'
"""Export model to ONNX format."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch


def main():
    """Export model to ONNX."""
    print("Model export script")
    print("Implement ONNX export logic here")


if __name__ == "__main__":
    main()
EOF

# ============================================
# 12. CONFIGS
# ============================================
echo -e "${YELLOW}Creating config files...${NC}"

cat > configs/train_config.yaml << 'EOF'
model:
  name: vit_base_patch16
  num_classes: 12
  pretrained: true

training:
  batch_size: 16
  epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.0001
  early_stopping_patience: 10

data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  image_size: 512
  num_workers: 4

optimizer:
  type: adamw
  lr: 0.0001
  weight_decay: 0.0001
EOF

cat > configs/inference_config.yaml << 'EOF'
model:
  checkpoint_path: "checkpoints/best_model.pth"
  num_classes: 12

inference:
  batch_size: 1
  image_size: 512
  device: "cuda"
EOF

cat > configs/mlflow_config.yaml << 'EOF'
mlflow:
  tracking_uri: "./mlruns"
  experiment_name: "{{cookiecutter.project_slug}}"
  
logging:
  log_params: true
  log_metrics: true
  log_models: true
EOF

# ============================================
# 13. TESTS
# ============================================
echo -e "${YELLOW}Creating tests...${NC}"

touch tests/__init__.py

cat > tests/test_data.py << 'EOF'
"""Tests for data module."""
import pytest
import numpy as np


def test_dataset_creation():
    """Test dataset creation."""
    assert True  # Placeholder


def test_data_augmentation():
    """Test data augmentation."""
    assert True  # Placeholder
EOF

cat > tests/test_model.py << 'EOF'
"""Tests for model module."""
import pytest
import torch


def test_model_forward():
    """Test model forward pass."""
    assert True  # Placeholder


def test_loss_functions():
    """Test loss functions."""
    assert True  # Placeholder
EOF

cat > tests/test_inference.py << 'EOF'
"""Tests for inference module."""
import pytest


def test_predictor():
    """Test predictor."""
    assert True  # Placeholder
EOF

# ============================================
# 14. DOCKER
# ============================================
echo -e "${YELLOW}Creating Docker files...${NC}"

cat > docker/Dockerfile << 'EOF'
FROM pytorch/pytorch:2.0.1-cuda{{cookiecutter.cuda_version}}-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/base.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

RUN pip install -e .

CMD ["python", "scripts/train.py"]
EOF

cat > docker/docker-compose.yml << 'EOF'
version: '3.8'

services:
  training:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    volumes:
      - ../data:/app/data
      - ../models:/app/models
      - ../logs:/app/logs
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
EOF

# ============================================
# 15. REQUIREMENTS
# ============================================
echo -e "${YELLOW}Creating requirements files...${NC}"

cat > requirements/base.txt << 'EOF'
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
albumentations>=1.3.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
pyyaml>=6.0
tqdm>=4.65.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
pandas>=2.0.0
scipy>=1.10.0
EOF

cat > requirements/dev.txt << 'EOF'
-r base.txt
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.7.0
flake8>=6.1.0
isort>=5.12.0
jupyter>=1.0.0
ipykernel>=6.25.0
mlflow>=2.5.0
tensorboard>=2.13.0
EOF

cat > requirements/prod.txt << 'EOF'
-r base.txt
gunicorn>=21.2.0
fastapi>=0.100.0
uvicorn>=0.23.0
EOF

# ============================================
# 16. NOTEBOOKS
# ============================================
echo -e "${YELLOW}Creating notebooks...${NC}"

cat > notebooks/00_eda_analysis.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# ⭐ Exploratory Data Analysis - First Step\n",
    "\n",
    "This notebook performs comprehensive EDA on the floorplan dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.insert(0, '../src')\n",
    "\n",
    "from eda.dataset_analysis import analyze_dataset_structure\n",
    "from eda.class_distribution import analyze_class_distribution\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run EDA analysis\n",
    "print(\"Dataset Analysis\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

cat > notebooks/01_preprocessing.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Preprocessing notebook\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

cat > notebooks/02_model_training.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Training notebook\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

cat > notebooks/03_evaluation.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Evaluation notebook\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# ============================================
# 17. OTHER FILES
# ============================================
echo -e "${YELLOW}Creating other essential files...${NC}"

cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
dist/
*.egg-info/

# Jupyter
.ipynb_checkpoints

# Data
data/raw/
data/processed/
*.csv
*.h5

# Models
models/*.pth
models/*.pt
*.onnx

# Logs
logs/
*.log
mlruns/

# Environment
.env
.venv

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
EOF

cat > .env.example << 'EOF'
# Project Configuration
PROJECT_NAME={{cookiecutter.project_slug}}
PYTHON_VERSION={{cookiecutter.python_version}}

# Data Paths
DATA_DIR=./data
RAW_DATA_DIR=./data/raw
PROCESSED_DATA_DIR=./data/processed

# MLFlow
MLFLOW_TRACKING_URI=./mlruns
MLFLOW_EXPERIMENT_NAME={{cookiecutter.project_slug}}

# Training
CUDA_VISIBLE_DEVICES=0
NUM_WORKERS=4
BATCH_SIZE=16
EOF

cat > setup.py << 'EOF'
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="{{cookiecutter.project_slug}}",
    version="{{cookiecutter.version}}",
    author="{{cookiecutter.author_name}}",
    author_email="{{cookiecutter.author_email}}",
    github_username="{{cookiecutter.github_username}}",
    description="{{cookiecutter.description}}",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.9",
)
EOF

cat > Makefile << 'EOF'
.PHONY: help install dev-install test lint format clean eda train docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  dev-install  - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  clean        - Clean generated files"
	@echo "  eda          - Run EDA analysis (FIRST STEP ⭐)"
	@echo "  train        - Start training"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"

install:
	pip install -r requirements/base.txt
	pip install -e .

dev-install:
	pip install -r requirements/dev.txt
	pip install -e .

test:
	pytest tests/ -v --cov=src

lint:
	flake8 src/ --max-line-length=120
	isort --check-only src/

format:
	black src/
	isort src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info

eda:
	python scripts/eda_analysis.py

train:
	python scripts/train.py

docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-run:
	docker-compose -f docker/docker-compose.yml up
EOF

cat > README.md << 'EOF'
# {{cookiecutter.project_name}}

{{cookiecutter.description}}

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/{{cookiecutter.github_username}}/{{cookiecutter.project_slug}}.git
cd {{cookiecutter.project_slug}}

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make dev-install
```

### ⭐ First Step: Run EDA Analysis
```bash
make eda
```

Or run directly:
```bash
python scripts/eda_analysis.py
```

### Training
```bash
make train
```

## 📁 Project Structure
```
{{cookiecutter.project_slug}}/
├── src/              # Source code
├── notebooks/        # Jupyter notebooks
├── tests/            # Unit tests
├── configs/          # Configuration files
├── scripts/          # Training and utility scripts
└── docker/           # Docker configuration
```

## 🔧 Development
```bash
# Run tests
make test

# Format code
make format

# Lint code
make lint
```

## 🐳 Docker
```bash
# Build Docker image
make docker-build

# Run container
make docker-run
```

## 📊 MLFlow Tracking
```bash
mlflow ui
```

## 📝 License

{{cookiecutter.license}}

## 👤 Author

**{{cookiecutter.author_name}}**
- Email: {{cookiecutter.author_email}}
- GitHub: [@{{cookiecutter.github_username}}](https://github.com/{{cookiecutter.github_username}})
EOF

# Create empty mlruns directory
mkdir -p mlruns
cat > mlruns/.gitkeep << 'EOF'
# MLFlow experiments directory
EOF

# Make scripts executable
chmod +x scripts/*.py

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Template created successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Template location:${NC} $TEMPLATE_DIR"
echo ""
echo -e "${YELLOW}📝 To use the template:${NC}"
echo ""
echo "  1. Navigate to where you want to create your project:"
echo "     cd ~/projects"
echo ""
echo "  2. Run cookiecutter:"
echo "     cookiecutter $TEMPLATE_DIR"
echo ""
echo "  3. After generation, navigate to your new project:"
echo "     cd your-project-name"
echo ""
echo "  4. Set up and run EDA (First Step ⭐):"
echo "     python -m venv venv"
echo "     source venv/bin/activate"
echo "     make dev-install"
echo "     make eda"
echo ""
echo -e "${GREEN}========================================${NC}"