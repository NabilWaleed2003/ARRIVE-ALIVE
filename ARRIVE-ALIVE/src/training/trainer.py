"""Training utilities for the drowsiness detection model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Tuple
from src.utils import DEVICE, SEED, MODEL_CHECKPOINT_PATH


class Trainer:
    """Trainer class for model training and checkpoint management."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer,
        device: torch.device = DEVICE,
    ):
        """Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            criterion: Loss function
            optimizer: Optimizer instance
            device: Device to train on (CPU or CUDA)
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.losses = []

    def train_epoch(self, train_dataloader: DataLoader) -> float:
        """Train for one epoch.
        
        Args:
            train_dataloader: DataLoader for training data
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        
        for images, labels in train_dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        self.losses.append(avg_loss)
        
        return avg_loss

    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int
    ) -> List[float]:
        """Train the model for multiple epochs.
        
        Args:
            train_dataloader: DataLoader for training data
            num_epochs: Number of epochs to train
            
        Returns:
            List of average losses per epoch
        """
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(train_dataloader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        
        return self.losses

    def save_checkpoint(self, filepath: str = MODEL_CHECKPOINT_PATH):
        """Save model checkpoint.
        
        Args:
            filepath: Path to save the checkpoint
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
        }, filepath)
        print(f'Checkpoint saved to {filepath}')

    def load_checkpoint(self, filepath: str = MODEL_CHECKPOINT_PATH):
        """Load model checkpoint.
        
        Args:
            filepath: Path to the checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.losses = checkpoint.get('losses', [])
        print(f'Checkpoint loaded from {filepath}')


def create_trainer(
    model: nn.Module,
    learning_rate: float = 0.001,
    device: torch.device = DEVICE
) -> Trainer:
    """Create a trainer with default settings.
    
    Args:
        model: PyTorch model to train
        learning_rate: Learning rate for optimizer
        device: Device to train on
        
    Returns:
        Trainer instance
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    trainer = Trainer(model, criterion, optimizer, device)
    
    return trainer
