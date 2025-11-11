![header](doc/imgs/LogoHeader.png)

# Floor Plan ViT Classifier

A Vision Transformer (ViT) based deep learning model for semantic segmentation of architectural floor plans. This project implements a state-of-the-art ViT architecture to classify and segment different room types and architectural elements in floor plan images using the CubiCasa5K dataset.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-orange.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Semantic Classes](#semantic-classes)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Pipeline](#pipeline)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Vision Transformer Architecture**: Custom ViT-Small model with encoder-decoder design
- **12 Semantic Classes**: Room-type semantic segmentation (Background, Walls, Kitchen, Living Room, Bedroom, Bathroom, Hallway, Storage, Garage, Undefined, Closet, Balcony)
- **SVG-to-PNG Converter**: Automatic conversion of CubiCasa5K SVG annotations to semantic segmentation masks
- **Class Imbalance Handling**: Weighted loss function and dynamic class weight calculation
- **CubiCasa5K Support**: Full pipeline support for the CubiCasa5K dataset (5000+ floor plans)
- **Mixed Precision Training**: Fast training with CUDA mixed precision (AMP)
- **Exploratory Data Analysis**: Built-in EDA tools for dataset analysis
- **Flexible Configuration**: YAML-based configuration system
- **Visualization Tools**: Rich visualization of predictions and metrics

## 🏗️ Architecture

The model uses a custom Vision Transformer architecture specifically designed for semantic segmentation:

- **Input**: 512×512 RGB floor plan images
- **Patch Embedding**: Converts images into 16×16 patches (32×32 pixels each)
- **Transformer Encoder**: 12-layer transformer with 6 attention heads
- **Transformer Decoder**: 3-layer decoder for upsampling
- **Segmentation Head**: Dense prediction layer for 12 semantic classes
- **Total Parameters**: ~84M trainable parameters

## 🎨 Semantic Classes

The model segments floor plans into 12 semantic classes:

| Class | Description | Color Code |
|-------|-------------|-----------|
| 0 | Background/Structural (walls boundaries, doors, windows) | Black |
| 1 | Walls | Dark Gray |
| 2 | Kitchen | Red |
| 3 | Living Room | Green |
| 4 | Bedroom | Blue |
| 5 | Bathroom | Yellow |
| 6 | Hallway/Entry Lobby | Cyan |
| 8 | Storage | Magenta |
| 9 | Garage | Orange |
| 10 | Undefined/Closets | Gray |
| 11 | Balcony/Outdoor | Light Blue |

## 📦 Requirements

- **Python**: 3.12+
- **CUDA**: 11.8+ (for GPU training)
- **GPU Memory**: 8GB+ recommended (tested with RTX 3090)
- **Storage**: ~100GB for CubiCasa5K dataset + preprocessing

### Core Dependencies

- PyTorch 2.5.1+ with CUDA support
- OpenCV 4.8+
- Pillow 10.0+
- NumPy, SciPy
- tqdm (progress bars)
- See `requirements/base.txt` for complete list

## 🚀 Installation

### Quick Start (Conda)

```bash
# Clone repository
git clone https://github.com/yourusername/floorplan-vit-classifier.git
cd floorplan-vit-classifier/VpC3

# Create conda environment
conda env create -f environment.yml
conda activate floorplan_vit

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Alternative: Pip Virtual Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements/base.txt

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

## 📊 Dataset Setup

### Step 1: Download CubiCasa5K

```bash
# The dataset will be automatically downloaded when needed
# Or manually: https://zenodo.org/record/4817057
# Expected path: ~/.cache/kagglehub/datasets/qmarva/cubicasa5k/versions/4/
```

### Step 2: Project Structure

After setup, your data directory should look like:

```
VpC3/data/
├── cubicasa5k_converted/       # SVG converter output
│   ├── images/                 # Original floor plan images
│   └── annotations/            # Semantic segmentation masks (PNG)
└── processed/                  # Preprocessed dataset (after preprocessing)
    ├── images/                 # Resized to 512x512
    └── annotations/            # Semantic masks (resized)
```

## 🔄 Pipeline

The complete workflow from raw data to training:

### 1. **SVG Conversion** (Convert SVG annotations to semantic masks)

```bash
cd VpC3
python src/data/svg_to_png_converter.py --test 1  # Test with 1 sample

# Full conversion
python src/data/svg_to_png_converter.py
```

Output:
- `data/cubicasa5k_converted/images/` - Floor plan PNG images
- `data/cubicasa5k_converted/annotations/` - Semantic segmentation masks (12 classes)

### 2. **Preprocessing** (Validate, normalize, and resize)

```bash
python scripts/run_preprocessing.py
```

Output:
- `data/processed/images/` - 512×512 resized images
- `data/processed/annotations/` - Corresponding semantic masks

### 3. **Training** (Train the ViT model)

```bash
python scripts/train.py
```

Checkpoints saved to: `models/checkpoints/`

### 4. **Inference** (Test on validation/test set)

```bash
python scripts/test_inference.py
```

Results saved to: `outputs/inference_results/`

## 🎯 Usage

### Option A: Full Pipeline (Recommended)

```bash
cd VpC3

# 1. Convert SVG annotations to semantic masks
python src/data/svg_to_png_converter.py

# 2. Preprocess the dataset
python scripts/run_preprocessing.py

# 3. Run EDA on preprocessed data
python src/eda/eda_analysis.py --dataset_path data/processed

# 4. Train the model
python scripts/train.py

# 5. Run inference
python scripts/test_inference.py
```

### Option B: Quick Test

```bash
cd VpC3

# Convert only 5 samples for testing
python src/data/svg_to_png_converter.py --test 5

# Preprocess
python scripts/run_preprocessing.py

# Train with fewer epochs
python scripts/train.py
```

### Training Parameters

Edit configuration before training:

```python
# In scripts/train.py, modify CONFIG:
CONFIG = {
    'images_dir': 'data/processed/images',
    'masks_dir': 'data/processed/annotations',
    'batch_size': 4,              # Adjust for GPU memory
    'num_epochs': 50,             # Training epochs
    'learning_rate': 5e-5,        # Lower LR for stability
    'n_classes': 12,              # Semantic classes
    'use_class_weights': True,    # Handle class imbalance
    'mixed_precision': True,      # Faster training
}
```

### Monitor Training

```bash
# View training logs
tail -f VpC3/logs/*.log

# Check GPU usage
nvidia-smi -l 1  # Update every 1 second
```

## 📁 Project Structure

```
floorplan-vit-classifier/
└── VpC3/
    ├── src/
    │   ├── data/
    │   │   ├── svg_to_png_converter.py    # ✨ SVG to semantic mask conversion
    │   │   ├── dataset.py                 # PyTorch dataset classes
    │   │   └── preprocessing.py           # Data preprocessing utilities
    │   ├── models/
    │   │   └── vit_segmentation.py       # ViT architecture
    │   ├── eda/
    │   │   ├── eda_analysis.py           # Exploratory data analysis
    │   │   └── visualization.py          # Visualization tools
    │   ├── inference/
    │   │   └── inference_results/        # Prediction outputs
    │   └── utils/
    │       └── logging_config.py         # Logging utilities
    ├── scripts/
    │   ├── train.py                      # Main training script
    │   ├── run_preprocessing.py          # Preprocessing pipeline
    │   ├── run_dataset.py                # Dataset testing
    │   ├── test_inference.py             # Inference script
    │   └── diagnose_model.py             # Model diagnostics
    ├── data/
    │   ├── cubicasa5k_converted/         # Converted SVG → PNG
    │   └── processed/                    # Preprocessed dataset
    ├── models/
    │   └── checkpoints_fixed/            # Model checkpoints
    ├── outputs/
    │   └── eda/                          # EDA analysis outputs
    ├── configs/
    │   ├── config.yaml                   # Main configuration
    │   └── class_mapping_256_to_34.json  # Legacy mapping
    ├── requirements/
    │   ├── base.txt                      # Core dependencies
    │   ├── dev.txt                       # Development dependencies
    │   └── prod.txt                      # Production dependencies
    ├── environment.yml                   # Conda environment
    └── README.md                         # This file
```

## ⚙️ Configuration

### Training Configuration

Key parameters in `scripts/train.py`:

```python
CONFIG = {
    # Data
    'images_dir': 'data/processed/images',
    'masks_dir': 'data/processed/annotations',
    'batch_size': 4,
    'num_workers': 0,  # Change to 4 on Linux/Mac
    
    # Model - ViT-Small
    'img_size': 512,
    'patch_size': 32,
    'n_classes': 12,                      # 12 semantic classes
    'embed_dim': 384,
    'n_encoder_layers': 12,
    'n_decoder_layers': 3,
    'n_heads': 6,
    'mlp_ratio': 4.0,
    'dropout': 0.1,
    
    # Training
    'num_epochs': 50,
    'learning_rate': 5e-5,                # Lower LR for weighted loss
    'weight_decay': 0.01,
    'mixed_precision': True,              # AMP for faster training
    
    # Loss & Optimization
    'use_class_weights': True,            # Handle class imbalance
    'label_smoothing': 0.1,               # Regularization
    
    # Checkpointing
    'checkpoint_dir': 'models/checkpoints_fixed',
    'save_frequency': 10
}
```

### Class Weight Calculation

Automatic class weights based on inverse frequency:

```python
# Calculated during training startup
class_weights = calculate_class_weights(train_loader, num_classes=12)
# Minority classes get higher weights
# Helps model learn underrepresented room types
```

## 📈 Results

### Expected Performance

On CubiCasa5K validation set:

| Metric | Expected Range |
|--------|-----------------|
| Mean IoU | 0.60-0.75 |
| Pixel Accuracy | 0.80-0.90 |
| Training Time | 8-12 hours (RTX 3090) |

### Per-Class Performance

- **Well-segmented**: Walls, Living Rooms, Bedrooms, Kitchens
- **Moderate**: Bathrooms, Hallways
- **Challenging**: Storage, Undefined, Small rooms (due to class imbalance)

### Training Artifacts

Training generates:

- `models/checkpoints/best_model.pth` - Best validation IoU
- `models/checkpoints/final_model.pth` - Final trained model
- `models/checkpoints/training_history.json` - Loss & metrics history
- `outputs/eda/` - Dataset analysis visualizations

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```python
# In train.py, reduce batch size:
CONFIG['batch_size'] = 2  # or 1

# Enable gradient accumulation:
# (Implement in training loop for effective batch size)
```

#### 2. Import Errors

```bash
# Ensure running from VpC3 directory
cd VpC3

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%cd%           # Windows CMD
```

#### 3. Dataset Not Found

```bash
# Verify dataset structure
ls -la data/processed/

# Check file counts
find data/processed/images -type f | wc -l
find data/processed/annotations -type f | wc -l

# Should be equal numbers
```

#### 4. Poor Model Performance

**Possible causes and solutions:**

- **Class imbalance**: Already handled with weighted loss ✓
- **Low learning rate**: Try 1e-4 or 5e-5
- **Insufficient epochs**: Train for 100+ epochs
- **Data quality**: Check EDA results
- **Model initialization**: Ensure pretrained weights load correctly

### Performance Optimization

**For faster training:**

1. ✓ Mixed precision enabled by default
2. ✓ Optimal batch size (4) pre-configured
3. Increase `num_workers` on multi-core machines
4. Use SSD for dataset storage (faster I/O)

**For better results:**

1. ✓ Class weights automatically calculated
2. ✓ Label smoothing enabled
3. Longer training (100+ epochs)
4. Data augmentation (in dataset.py)
5. Learning rate scheduling (cosine annealing with restarts)

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch: `git checkout -b feature/improvement`
3. Commit changes: `git commit -m 'Add improvement'`
4. Push: `git push origin feature/improvement`
5. Create Pull Request


## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CubiCasa5K Dataset](https://zenodo.org/record/4817057)
- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
- PyTorch and OpenCV communities

## 📞 Support

- **Issues**: GitHub Issues
- **Questions**: GitHub Discussions
- **Email**: support@yourproject.com

---

**Made with ❤️ for the architecture and computer vision communities**

*Last Updated: November 11, 2025*

![footer](doc/imgs/LogoFooter.png)
