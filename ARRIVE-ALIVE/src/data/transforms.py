"""Image transformation utilities for the drowsiness detection dataset."""

from torchvision import transforms
from src.utils import MEAN, STD, IMAGE_SIZE


def get_base_transform():
    """Get basic image transformation without normalization.
    
    Returns:
        torchvision.transforms.Compose: Composition of transforms
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])


def get_normalized_transform():
    """Get image transformation with normalization.
    
    Uses pre-calculated mean and standard deviation values.
    
    Returns:
        torchvision.transforms.Compose: Composition of transforms with normalization
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


def get_train_transform():
    """Get training-specific transforms with augmentation.
    
    Returns:
        torchvision.transforms.Compose: Composition of transforms with augmentation
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


def get_eval_transform():
    """Get evaluation-specific transforms (no augmentation).
    
    Returns:
        torchvision.transforms.Compose: Composition of evaluation transforms
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
