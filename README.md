![header](doc/imgs/LogoHeader.png)

# Floor Plan Analysis Hub

This repository serves as a centralized index for various Computer Vision models implemented for semantic segmentation and architectural floor plan analysis. The project explores and compares different state-of-the-art architectures using the **CubiCasa5K** dataset.

> **Note:** This repository functions as a hub. The source code, training scripts, and detailed technical documentation for each model reside in their respective branches.

### Implemented Architectures

We currently maintain two main approaches for floor plan segmentation:

| Model / Branch | Status |
| :--- | :--- |
| **[Vision Transformer (ViT)](https://github.com/BenjaSar/VpC3/tree/vit_final)** | ✅ **Completed** | 
| **[UNet++ Improved](https://github.com/BenjaSar/VpC3/tree/unet_plus_plus_improved)** | ✅ **Completed** | 
<!--🛠 **In Development**--->


[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Navigation & Usage](#-navigation--usage)
- [Architectures Description](#architectures-description)
- [Dataset Setup](#-dataset-setup)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Results](#-results)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)
- [Roadmap](#-roadmap)

<!-- 
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
- See `requirements/base.txt` for complete list -->

## 🚀 Navigation & Usage

To work with a specific architecture, clone the repository and switch to the corresponding branch:

### 1. Clone the repository
```bash
git clone [https://github.com/BenjaSar/VpC3.git](https://github.com/BenjaSar/VpC3.git)
cd VpC3
```
### 2. Select the model
#### To use the Vision Transformer:
```bash
git checkout vit_final
# You will now see the detailed README and training scripts for ViT
```
#### To use the UNet++:
```bash
git checkout unet_plus_plus_improved
# Switches to the improved convolutional architecture code
```

## Architectures Description

### **[Vision Transformer (ViT)](https://github.com/BenjaSar/VpC3/tree/vit_final)** <!-- - *Global Approach* -->
Uses a custom *ViT-Small* architecture with an Encoder-Decoder design.
- Splits image into 16x16 patches (embedding).
- **Encoder:** 12 Transformer layers with Self-Attention to capture global context.
- **Decoder:** 3 layers to recover spatial resolution.
- Segments **34 classes** (walls, rooms, openings).

### **[UNet++ Improved](https://github.com/BenjaSar/VpC3/tree/unet_plus_plus_improved)** <!-- - *Local & Edge Approach* -->
An evolution of U-Net with dense, nested connections (*Nested Skip Pathways*).
- Reduces the semantic gap between encoder and decoder feature maps.
- Implements **Deep Supervision** to improve gradient flow.
- Ideal for improving edge precision on fine architectural elements. |

## 📊 Dataset Setup

All models are designed to work with CubiCasa5K, a large-scale dataset containing 5000 floor plans with annotations for 80 different categories.

***Reference:*** [CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis](https://github.com/CubiCasa/CubiCasa5k)

### Supported Classes (34 total):

```
Background, Outdoor, Wall, Kitchen, Living Room, Bedroom, Bath,
Entry, Railing, Storage, Garage, Undefined, Interior Door,
Exterior Door, Window, and more...
```

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
```bash
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

### Exploratory Data Analysis (EDA)

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
  author={Grupo 3 VpC},
  year={2025},
  url={https://github.com/BenjaSar/floorplan-vit-classifier}
}
```

### Related Papers
- **CubiCasa5K**: [CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis](https://arxiv.org/abs/1904.01920)
- **DeiT**: [Training data-efficient image transformers](https://arxiv.org/abs/2012.12877)
- **Vision Transformer**: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- **Unet Plus Plus**: [UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/abs/1807.10165)

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) for the dataset
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) for model implementations
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [MLflow](https://mlflow.org/) for experiment tracking

## 📞 Contact

For questions or issues:

- **GitHub Issues**: [Create an issue](https://github.com/BenjaSar/VpC3/issues)
<!-- - **Discussion**: [GitHub Discussions](https://github.com/BenjaSar/VpC3/discussions) -->

## 🗺️ Roadmap

- [x] Add pre-trained model weights
- [x] Implement ViT-Base and ViT-Large variants
- [ ] Support for additional datasets (R-FID, LIFULL)
- [x] Web-based inference demo
- [ ] Docker containerization
- [ ] Model quantization for deployment
- [ ] Real-time inference optimization
- [ ] Integration with architectural CAD software

---

**Made with ❤️ for the computer vision and architecture communities**

*Last Updated: November 20, 2025*

![footer](doc/imgs/LogoFooter.png)