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
