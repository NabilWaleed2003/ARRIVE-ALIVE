"""Dataset utilities for the drowsiness detection project."""

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Tuple
from src.utils import (
    DATA_DIR,
    DROWSY_DIR,
    NON_DROWSY_DIR,
    TRAIN_TEST_SPLIT,
    BATCH_SIZE,
)


class DrownsinessDataset(Dataset):
    """Custom dataset for drowsiness detection.
    
    Loads images from Drowsy and Non-Drowsy directories and provides
    corresponding labels.
    """

    def __init__(self, data_paths: List[str], transform=None):
        """Initialize the dataset.
        
        Args:
            data_paths: List of image file paths
            transform: Optional image transformations
        """
        self.data_paths = data_paths
        self.transform = transform
        self.labels = [1 if 'Drowsy' in path else 0 for path in data_paths]

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.data_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample and its label.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image_tensor, label_tensor)
        """
        img_path = self.data_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        label = 1 if 'Drowsy' in img_path else 0
        label = torch.tensor(label, dtype=torch.long)
        
        return img, label


def load_dataset_paths(data_dir: str = DATA_DIR) -> Tuple[List[str], List[str]]:
    """Load dataset paths and split into train and test sets.
    
    Args:
        data_dir: Path to the main dataset directory
        
    Returns:
        Tuple of (train_paths, test_paths)
    """
    drowsy_paths = [
        os.path.join(data_dir, DROWSY_DIR, p)
        for p in os.listdir(os.path.join(data_dir, DROWSY_DIR))
    ]
    non_drowsy_paths = [
        os.path.join(data_dir, NON_DROWSY_DIR, p)
        for p in os.listdir(os.path.join(data_dir, NON_DROWSY_DIR))
    ]
    
    all_paths = drowsy_paths + non_drowsy_paths
    random.shuffle(all_paths)
    
    split_idx = int(TRAIN_TEST_SPLIT * len(all_paths))
    train_paths = all_paths[:split_idx]
    test_paths = all_paths[split_idx:]
    
    return train_paths, test_paths


def create_dataloaders(
    train_paths: List[str],
    test_paths: List[str],
    train_transform=None,
    eval_transform=None,
    batch_size: int = BATCH_SIZE
) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for training and evaluation.
    
    Args:
        train_paths: List of training image paths
        test_paths: List of test image paths
        train_transform: Transform for training data
        eval_transform: Transform for evaluation data
        batch_size: Batch size for data loaders
        
    Returns:
        Tuple of (train_dataloader, test_dataloader)
    """
    train_dataset = DrownsinessDataset(train_paths, transform=train_transform)
    test_dataset = DrownsinessDataset(test_paths, transform=eval_transform)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_dataloader, test_dataloader


def calculate_mean_std(dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate mean and standard deviation of the dataset.
    
    Args:
        dataloader: DataLoader to compute statistics from
        
    Returns:
        Tuple of (mean, std) tensors
    """
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    for images, _ in dataloader:
        batch_size, channels, height, width = images.shape
        total_images += batch_size
        mean += images.mean([0, 2, 3]) * batch_size
        std += images.std([0, 2, 3]) * batch_size

    mean /= total_images
    std /= total_images
    
    return mean, std
