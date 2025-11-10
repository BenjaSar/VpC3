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
