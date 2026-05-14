"""Convolutional Neural Network model for drowsiness detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import NUM_CLASSES, DROPOUT_RATE


class DrowsinessDetectionCNN(nn.Module):
    """Convolutional Neural Network for drowsiness detection.
    
    A 4-layer CNN architecture with batch normalization and dropout
    for robust feature extraction and classification.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout_rate: float = DROPOUT_RATE):
        """Initialize the CNN model.
        
        Args:
            num_classes: Number of output classes (default: 2)
            dropout_rate: Dropout rate for regularization (default: 0.3)
        """
        super(DrowsinessDetectionCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Activation and pooling
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Conv block 1
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Conv block 4
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = self.flatten(x)
        
        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x


def create_model(num_classes: int = NUM_CLASSES) -> DrowsinessDetectionCNN:
    """Create and return a drowsiness detection model.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        DrowsinessDetectionCNN model instance
    """
    return DrowsinessDetectionCNN(num_classes=num_classes)
