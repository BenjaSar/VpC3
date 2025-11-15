"""
Project Initialization Script
Sets up the complete project structure, configurations, and environment
"""

# import os
import sys
# import json
# import shutil
# import logging
from pathlib import Path
# from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging

logger = setup_logging()


def create_directory_structure(project_root: Path):
    """
    Create complete project directory structure
    
    Args:
        project_root: Root project directory
    """
    logger.info("Creating directory structure...")
    
    directories = [
        "data/raw",
        "data/processed",
        "data/splits",
        "models/checkpoints",
        "models/final",
        "models/quantized",
        "logs",
        "eda_output",
        "notebooks",
        "scripts",
        "src/config",
        "src/data",
        "src/models",
        "src/training",
        "src/inference",
        "src/utils",
        "src/eda",
        "tests",
        "docker",
        "configs",
        "mlruns",
        ".dvc",
        ".git"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created: {directory}")
    
    logger.info("Directory structure created successfully!")


def create_env_file(project_root: Path, config: dict):
    """
    Create .env file with project configuration
    
    Args:
        project_root: Root project directory
        config: Configuration dictionary
    """
    logger.info("Creating .env file...")
    
    env_content = f"""# Project Configuration
PROJECT_NAME={config['project_name']}
PROJECT_VERSION=0.1.0
PROJECT_ROOT={project_root}

# Dataset Configuration
DATASET_PATH=./data/raw
PROCESSED_DATA_PATH=./data/processed
DATASET_TYPE={config['dataset_type']}
NUM_CLASSES={config['num_classes']}

# Model Configuration
MODEL_NAME=vit-small-segmentation
PRETRAINED_MODEL=facebook/deit-small-patch16-224
IMAGE_SIZE={config['image_size']}
PATCH_SIZE=16
EMBEDDING_DIM=384
NUM_HEADS=6
NUM_LAYERS=12
DECODER_DIM=256

# Training Configuration
BATCH_SIZE=8
NUM_EPOCHS=50
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-5
WARMUP_STEPS=500
GRADIENT_ACCUMULATION_STEPS=1
MIXED_PRECISION=True
NUM_WORKERS=4
SEED=42

# GPU Configuration
DEVICE=cuda
GPU_MEMORY={config['gpu_memory']}
NUM_GPUS=1

# MLFlow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=floorplan-vit-experiments
MLFLOW_BACKEND_STORE_URI=./mlruns

# Logging Configuration
LOG_LEVEL=INFO
LOG_PATH=./logs

# Paths
CHECKPOINT_DIR=./models/checkpoints
FINAL_MODEL_DIR=./models/final
QUANTIZED_MODEL_DIR=./models/quantized
"""
    
    env_path = project_root / ".env"
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    logger.info(f"✓ Created: .env")


def create_main_config(project_root: Path, config: dict):
    """
    Create main configuration YAML file
    
    Args:
        project_root: Root project directory
        config: Configuration dictionary
    """
    logger.info("Creating main configuration file...")
    
    config_content = f"""# Main Configuration File
project:
  name: {config['project_name']}
  version: 0.1.0
  description: Vision Transformer for Floor Plan Room Classification
  root_dir: {project_root}

dataset:
  name: {config['dataset_type']}
  path: ./data/raw
  processed_path: ./data/processed
  num_classes: {config['num_classes']}
  split_ratio: [0.7, 0.15, 0.15]
  image_size: {config['image_size']}
  augmentation: true
  num_workers: 4

model:
  name: vit-small-segmentation
  backbone: facebook/deit-small-patch16-224
  pretrained: true
  num_classes: {config['num_classes']}
  patch_size: 16
  image_size: {config['image_size']}
  embedding_dim: 384
  num_heads: 6
  num_layers: 12
  decoder_dim: 256
  num_decoder_layers: 3
  dropout: 0.1

training:
  batch_size: 8
  num_epochs: 50
  learning_rate: 1.0e-4
  weight_decay: 1.0e-5
  warmup_steps: 500
  gradient_accumulation_steps: 1
  mixed_precision: true
  seed: 42
  device: cuda

optimizer:
  name: adamw
  beta1: 0.9
  beta2: 0.999
  epsilon: 1.0e-8

scheduler:
  name: cosine
  num_warmup_steps: 500
  num_training_steps: 50000

losses:
  segmentation:
    - name: focal_loss
      weight: 0.5
      alpha: 0.25
      gamma: 2.0
    - name: dice_loss
      weight: 0.5

metrics:
  - miou
  - dice
  - pixel_accuracy
  - per_class_iou

mlflow:
  tracking_uri: http://localhost:5000
  experiment_name: floorplan-vit-experiments
  backend_store_uri: ./mlruns

logging:
  level: INFO
  path: ./logs
  
paths:
  checkpoint_dir: ./models/checkpoints
  final_model_dir: ./models/final
  quantized_model_dir: ./models/quantized
  eda_output_dir: ./eda_output
  logs_dir: ./logs
"""
    
    config_path = project_root / "configs" / "config.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    logger.info(f"✓ Created: configs/config.yaml")


def create_data_pipeline_config(project_root: Path, config: dict):
    """
    Create data pipeline configuration
    
    Args:
        project_root: Root project directory
        config: Configuration dictionary
    """
    logger.info("Creating data pipeline configuration...")
    
    pipeline_content = f"""# Data Pipeline Configuration
data_pipeline:
  version: 1.0
  
  # Input configuration
  input:
    dataset_type: {config['dataset_type']}
    images_dir: images
    annotations_dir: annotations
    supported_formats: ['.jpg', '.jpeg', '.png', '.tiff']
  
  # Preprocessing
  preprocessing:
    resize:
      enabled: true
      size: {config['image_size']}
      method: bilinear
    normalize:
      enabled: true
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
    clip_values:
      enabled: true
      min: 0.0
      max: 1.0
  
  # Augmentation
  augmentation:
    enabled: true
    train_augmentations:
      - type: HorizontalFlip
        p: 0.5
      - type: VerticalFlip
        p: 0.5
      - type: Rotate
        limit: 15
        p: 0.5
      - type: ElasticTransform
        alpha: 1
        sigma: 50
        p: 0.3
      - type: RandomBrightnessContrast
        brightness_limit: 0.2
        contrast_limit: 0.2
        p: 0.5
      - type: GaussNoise
        p: 0.2
    
    val_augmentations:
      - type: Normalize
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
  
  # Data splitting
  splitting:
    strategy: stratified
    train_ratio: 0.7
    val_ratio: 0.15
    test_ratio: 0.15
    random_seed: 42
  
  # Validation
  validation:
    check_missing_files: true
    check_image_corrupted: true
    check_annotation_corrupted: true
    remove_invalid: false
    log_invalid: true
  
  # Output
  output:
    save_processed: true
    output_dir: ./data/processed
    format: npz
    compression: true
"""
    
    pipeline_path = project_root / "configs" / "data_pipeline.yaml"
    with open(pipeline_path, 'w') as f:
        f.write(pipeline_content)
    
    logger.info(f"✓ Created: configs/data_pipeline.yaml")


def create_training_config(project_root: Path):
    """
    Create training configuration
    
    Args:
        project_root: Root project directory
    """
    logger.info("Creating training configuration...")
    
    training_content = """# Training Configuration
training:
  # General settings
  project_name: floorplan-vit-classifier
  experiment_name: vit-small-segmentation
  run_name: baseline-run
  seed: 42
  deterministic: true
  
  # Device settings
  device: cuda
  num_gpus: 1
  mixed_precision: true
  
  # Training loop
  num_epochs: 50
  batch_size: 8
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  
  # Optimization
  optimizer:
    name: adamw
    lr: 1.0e-4
    weight_decay: 1.0e-5
    betas: [0.9, 0.999]
    eps: 1.0e-8
  
  scheduler:
    name: cosine
    warmup_steps: 500
    num_training_steps: 50000
    num_cycles: 0.5
  
  # Loss functions
  loss:
    primary: focal_loss
    secondary: dice_loss
    loss_weights:
      focal_loss: 0.5
      dice_loss: 0.5
    focal_loss_params:
      alpha: 0.25
      gamma: 2.0
  
  # Early stopping
  early_stopping:
    enabled: true
    metric: val_miou
    patience: 10
    min_delta: 0.001
    mode: max
  
  # Checkpointing
  checkpoint:
    enabled: true
    save_best: true
    save_last: true
    save_frequency: 5
    monitor_metric: val_miou
  
  # Validation
  validation:
    enabled: true
    frequency: 1
    compute_metrics: true
    metrics:
      - miou
      - dice
      - pixel_accuracy
      - per_class_iou
  
  # Logging
  logging:
    log_frequency: 100
    log_metrics: true
    log_gradients: false
    log_weights: false
  
  # MLFlow tracking
  mlflow:
    enabled: true
    track_params: true
    track_metrics: true
    track_artifacts: true
    log_model: true
  
  # Data loading
  dataloader:
    num_workers: 4
    pin_memory: true
    prefetch_factor: 2
    persistent_workers: true
"""
    
    training_path = project_root / "configs" / "training.yaml"
    with open(training_path, 'w') as f:
        f.write(training_content)
    
    logger.info(f"✓ Created: configs/training.yaml")


def create_inference_config(project_root: Path):
    """
    Create inference configuration
    
    Args:
        project_root: Root project directory
    """
    logger.info("Creating inference configuration...")
    
    inference_content = """# Inference Configuration
inference:
  # Model loading
  model:
    checkpoint_path: ./models/final/best_model.pth
    device: cuda
    dtype: float32
  
  # Batch processing
  batch_size: 16
  num_workers: 4
  pin_memory: true
  
  # Post-processing
  post_processing:
    apply_crf: false
    smooth_predictions: true
    min_region_size: 100
  
  # Output
  output:
    format: png
    save_probabilities: false
    save_attention_maps: false
    colorize_output: true
  
  # Performance
  optimization:
    use_onnx: false
    quantization: false
    quantization_type: int8
  
  # Logging
  logging:
    save_predictions: true
    save_errors: true
    verbose: true
"""
    
    inference_path = project_root / "configs" / "inference.yaml"
    with open(inference_path, 'w') as f:
        f.write(inference_content)
    
    logger.info(f"✓ Created: configs/inference.yaml")


def create_gitignore(project_root: Path):
    """
    Create .gitignore file
    
    Args:
        project_root: Root project directory
    """
    logger.info("Creating .gitignore...")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Data
data/raw/
data/processed/
*.zip
*.tar.gz

# Models
models/checkpoints/
models/final/
models/quantized/
*.pth
*.pt
*.onnx

# Logs
logs/
*.log

# MLFlow
mlruns/
.mlflow/

# DVC
.dvc/
.dvc.lock

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Testing
.pytest_cache/
.coverage
htmlcov/

# Environment
.env
.env.local

# OS
.DS_Store
Thumbs.db

# Temporary
*.tmp
*.temp
.cache/
"""
    
    gitignore_path = project_root / ".gitignore"
    with open(gitignore_path, 'w') as f:
        f.write(gitignore_content)
    
    logger.info(f"✓ Created: .gitignore")


def main():
    """
    Main initialization function
    """
    
    print_separator = lambda title: logger.info(f"\n{'='*80}\n{title.center(80)}\n{'='*80}\n")
    
    print_separator("PHASE 1: PROJECT INITIALIZATION")
    
    # Get project root
    project_root = Path(__file__).parent.parent
    logger.info(f"Project Root: {project_root}")