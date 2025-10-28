![header](doc/imgs/LogoHeader.png)

# Floor Plan ViT Classifier

A Vision Transformer (ViT) based deep learning model for semantic segmentation of architectural floor plans. This project implements a state-of-the-art ViT architecture to classify and segment different room types and architectural elements in floor plan images.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Vision Transformer Architecture**: Custom ViT-Small model with encoder-decoder design
- **34 Class Segmentation**: Supports detailed floor plan segmentation including rooms, walls, doors, windows, etc.
- **CubiCasa5K Support**: Compatible with the CubiCasa5K dataset (5000 floor plans)
- **Mixed Precision Training**: Fast training with CUDA mixed precision (AMP)
- **MLflow Integration**: Experiment tracking and model versioning
- **Comprehensive EDA**: Built-in exploratory data analysis tools
- **Flexible Configuration**: YAML-based configuration with Hydra/OmegaConf
- **Visualization Tools**: Rich visualization of predictions and metrics

## 🏗️ Architecture

The model uses a custom Vision Transformer architecture:

- **Patch Embedding**: Converts 512×512 images into 16×16 patches (32×32 pixels each)
- **Transformer Encoder**: 12-layer transformer with 6 attention heads
- **Transformer Decoder**: 3-layer decoder for semantic segmentation
- **Segmentation Head**: Upsampling layers to restore original image resolution
- **Parameters**: ~84M trainable parameters

### Supported Classes (34 total):

```
Background, Outdoor, Wall, Kitchen, Living Room, Bedroom, Bath,
Entry, Railing, Storage, Garage, Undefined, Interior Door,
Exterior Door, Window, and more...
```

## 📦 Requirements

- **Python**: 3.12+
- **CUDA**: 11.8+ (for GPU training)
- **GPU Memory**: 8GB+ recommended
- **Storage**: ~50GB for CubiCasa5K dataset

### Core Dependencies

- PyTorch 2.5.1 (CUDA 11.8)
- Transformers 4.43.0+
- PyTorch Lightning 2.3.0+
- OpenCV, Pillow, Matplotlib
- MLflow, Optuna, Hydra
- See `requirements/base.txt` for complete list

## 🚀 Installation

### Option 1: Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/floorplan-vit-classifier.git
cd floorplan-vit-classifier

# Create conda environment
conda env create -f environment.yml
conda activate floorplan_vit

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Option 2: Pip Virtual Environment

```bash
# Create virtual environment
python -m venv floorplan_vit
source floorplan_vit/bin/activate  # On Windows: floorplan_vit\Scripts\activate

# Install dependencies
pip install -r requirements/base.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Option 3: Development Installation

```bash
# Install with development dependencies
pip install -r requirements/dev.txt

# Install pre-commit hooks (optional)
pre-commit install
```

## 📊 Dataset Setup

### CubiCasa5K Dataset (Recommended)

1. **Download the dataset**:

```bash
# Option A: Clone from GitHub
git clone https://github.com/CubiCasa/CubiCasa5k.git data/cubicasa5k_raw

# Option B: Use the conversion script
python data/convert_cubicasa_proper.py
```

2. **Organize the dataset**:

The expected structure:
```
data/
├── cubicasa5k/
│   ├── images/              # Floor plan images
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   └── ...
│   └── annotations/         # Segmentation masks
│       ├── 0001.png
│       ├── 0002.png
│       └── ...
```

3. **Preprocess the dataset**:

```bash
# Run preprocessing pipeline
python run_preprocessing.py

# Or use the dataset script
python run_dataset.py
```

### Alternative Dataset Sources

See [DATASET_DOWNLOAD_GUIDE.md](varios/DATASET_DOWNLOAD_GUIDE.md) for:
- Roboflow datasets
- LIFULL HOME's dataset
- R-FID dataset
- Custom dataset creation

## 🎯 Usage

### 1. Exploratory Data Analysis (EDA)

Analyze your dataset before training:

```bash
python src/eda/eda_analysis.py \
    --dataset_path ./data/cubicasa5k \
    --dataset_type cubicasa5k \
    --output_dir ./eda_output
```

This generates:
- Image dimension distributions
- Class distribution analysis
- Pixel statistics
- Quality report
- Sample visualizations

### 2. Training

#### Quick Start (Default Configuration)

```bash
python train.py
```

#### Custom Configuration

```bash
# Modify configs/config.yaml, then run:
python train.py

# Or override specific parameters:
python train.py training.batch_size=16 training.num_epochs=100
```

#### Training Parameters

Key parameters in `configs/config.yaml`:

```yaml
training:
  batch_size: 8          # Adjust based on GPU memory
  num_epochs: 50         # Training epochs
  learning_rate: 1e-4    # Initial learning rate
  mixed_precision: true  # Enable AMP for faster training
```

#### Monitor Training

```bash
# View logs
tail -f logs/floorplan_vit_*.log

# Start MLflow UI
mlflow ui --port 5000
# Open http://localhost:5000 in browser
```

### 3. Inference

#### Test on Entire Test Set

```bash
python test_inference.py
```

Results will be saved to `inference_results/`:
- `test_metrics.json`: Overall metrics
- `visualizations/`: Prediction visualizations

#### Test Single Image

Modify `test_inference.py`:

```python
TEST_SINGLE_IMAGE = True
SINGLE_IMAGE_PATH = 'path/to/your/image.png'
```

Then run:

```bash
python test_inference.py
```

### 4. Custom Inference

```python
from train import ViTSegmentation
import torch
import cv2
import numpy as np

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('models/checkpoints/best_model.pth', map_location=device)

model = ViTSegmentation(
    img_size=512,
    patch_size=32,
    n_classes=34,
    embed_dim=384,
    n_encoder_layers=12,
    n_decoder_layers=3
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load and preprocess image
image = cv2.imread('your_image.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (512, 512))
image = image.astype(np.float32) / 255.0

# Normalize
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
image = (image - mean) / std

# Predict
image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
with torch.no_grad():
    output = model(image_tensor)
    pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

# pred is the segmentation mask
```

## 📁 Project Structure

```
floorplan-vit-classifier/
├── configs/
│   └── config.yaml              # Training configuration
├── data/
│   ├── dataset.py               # PyTorch dataset classes
│   ├── data.py                  # Data loading utilities
│   ├── conversion.py            # Dataset conversion scripts
│   └── cubicasa5k/              # Dataset directory
├── src/
│   ├── preprocessing.py         # Data preprocessing
│   ├── eda/                     # Exploratory data analysis
│   │   ├── eda_analysis.py
│   │   └── visualization.py
│   └── utils/
│       └── logging_config.py    # Logging utilities
├── models/
│   └── checkpoints/             # Saved model checkpoints
├── logs/                        # Training logs
├── inference_results/           # Inference outputs
├── requirements/
│   ├── base.txt                 # Core dependencies
│   ├── dev.txt                  # Development dependencies
│   └── prod.txt                 # Production dependencies
├── varios/
│   ├── DATASET_DOWNLOAD_GUIDE.md
│   └── TROUBLESHOOTING_LOW_IOU.md
├── train.py                     # Training script
├── test_inference.py            # Inference script
├── run_preprocessing.py         # Preprocessing pipeline
├── run_dataset.py               # Dataset setup
├── initialize_project.py        # Project initialization
└── README.md                    # This file
```

## ⚙️ Configuration

### Main Configuration (configs/config.yaml)

```yaml
project:
  name: floorplan-vit-classifier
  version: 0.1.0

dataset:
  name: cubicasa5k
  path: ./data
  num_classes: 34
  image_size: 512

model:
  name: vit-small-segmentation
  backbone: facebook/deit-small-patch16-224
  pretrained: true
  patch_size: 16
  embed_dim: 384
  n_encoder_layers: 12
  n_decoder_layers: 3

training:
  batch_size: 8
  num_epochs: 50
  learning_rate: 1.0e-4
  weight_decay: 1.0e-5
  mixed_precision: true
  num_workers: 4

optimizer:
  name: adamw
  beta1: 0.9
  beta2: 0.999

losses:
  segmentation:
    - name: focal_loss
      weight: 0.5
    - name: dice_loss
      weight: 0.5

metrics:
  - miou
  - dice
  - pixel_accuracy
  - per_class_iou
```

### Environment Variables (.env)

Create a `.env` file (see `.env.example`):

```bash
DATASET_PATH=./data/cubicasa5k
MLFLOW_TRACKING_URI=http://localhost:5000
CUDA_VISIBLE_DEVICES=0
```

## 📈 Results

### Expected Performance

On CubiCasa5K test set:

| Metric | Value |
|--------|-------|
| Mean IoU | 0.65-0.75 |
| Pixel Accuracy | 0.85-0.90 |
| Training Time | ~8-12 hours (RTX 3090) |

### Model Checkpoints

The training script saves:

- `best_model.pth`: Best model based on validation IoU
- `final_model.pth`: Model after final epoch
- `checkpoint_epoch_N.pth`: Periodic checkpoints (every 5 epochs)
- `training_history.json`: Loss and metric history

### Visualizations

Training generates:

- Loss curves (train/validation)
- IoU progression
- Learning rate schedule
- Sample predictions

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Solution**: Reduce batch size in `configs/config.yaml`:

```yaml
training:
  batch_size: 4  # or 2
  gradient_accumulation_steps: 2  # To maintain effective batch size
```

#### 2. Dataset Loading Errors

```bash
# Verify dataset structure
python -c "from data.dataset import FloorPlanDataset; FloorPlanDataset('data/cubicasa5k/images', 'data/cubicasa5k/annotations')"

# Run EDA to check data quality
python src/eda/eda_analysis.py --dataset_path ./data/cubicasa5k
```

#### 3. Import Errors

```bash
# Ensure project root is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%cd%          # Windows CMD
```

#### 5. MLflow Connection Issues

```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000

# Or use local tracking
# In configs/config.yaml:
mlflow:
  tracking_uri: ./mlruns  # Local directory
```

### Performance Optimization

**For faster training:**

1. Enable mixed precision (default)
2. Use multiple GPUs (if available)
3. Increase `num_workers` in dataloader
4. Use SSD for dataset storage
5. Profile with PyTorch profiler

**For better results:**

1. Use data augmentation
2. Adjust class weights for imbalanced classes
3. Experiment with different loss combinations
4. Use learning rate scheduling
5. Train longer (100+ epochs)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements/dev.txt

# Run tests
pytest tests/

# Format code
black src/ data/ *.py
ruff check src/ data/ *.py

# Type checking
mypy src/
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{floorplan_vit_classifier,
  title={Floor Plan Vision Transformer Classifier},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/floorplan-vit-classifier}
}
```

### Related Papers

- **Vision Transformer**: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- **CubiCasa5K**: [CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis](https://arxiv.org/abs/1904.01920)
- **DeiT**: [Training data-efficient image transformers](https://arxiv.org/abs/2012.12877)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) for the dataset
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) for model implementations
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [MLflow](https://mlflow.org/) for experiment tracking

## 📞 Contact

For questions or issues:

- **Email**: your.email@example.com
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/floorplan-vit-classifier/issues)
- **Discussion**: [GitHub Discussions](https://github.com/yourusername/floorplan-vit-classifier/discussions)

## 🗺️ Roadmap

- [ ] Add pre-trained model weights
- [ ] Implement ViT-Base and ViT-Large variants
- [ ] Support for additional datasets (R-FID, LIFULL)
- [ ] Web-based inference demo
- [ ] Docker containerization
- [ ] Model quantization for deployment
- [ ] Real-time inference optimization
- [ ] Integration with architectural CAD software

---

**Made with ❤️ for the computer vision and architecture communities**

*Last Updated: October 28, 2025*

![footer](doc/imgs/LogoFooter.png)

