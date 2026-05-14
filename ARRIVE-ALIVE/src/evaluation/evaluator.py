"""Evaluation utilities for model testing and metrics."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict
from src.utils import DEVICE, CLASS_NAMES


class Evaluator:
    """Evaluator class for model evaluation and metric computation."""

    def __init__(self, model: nn.Module, device: torch.device = DEVICE):
        """Initialize the evaluator.
        
        Args:
            model: PyTorch model to evaluate
            device: Device to evaluate on (CPU or CUDA)
        """
        self.model = model
        self.device = device

    def evaluate(self, test_dataloader: DataLoader) -> Tuple[float, Dict]:
        """Evaluate model on test dataset.
        
        Args:
            test_dataloader: DataLoader for test data
            
        Returns:
            Tuple of (accuracy, metrics_dict)
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        metrics = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
        }
        
        return accuracy, metrics

    def predict(self, image: torch.Tensor) -> str:
        """Predict class for a single image.
        
        Args:
            image: Input image tensor
            
        Returns:
            Predicted class name
        """
        self.model.eval()
        with torch.no_grad():
            image = image.to(self.device)
            if image.dim() == 3:  # Add batch dimension if needed
                image = image.unsqueeze(0)
            
            output = self.model(image)
            prediction = torch.argmax(output, dim=1).item()
        
        return CLASS_NAMES.get(prediction, "Unknown")


def evaluate_model(
    model: nn.Module,
    test_dataloader: DataLoader,
    device: torch.device = DEVICE
) -> Tuple[float, Dict]:
    """Evaluate a model on test data.
    
    Args:
        model: PyTorch model to evaluate
        test_dataloader: DataLoader for test data
        device: Device to evaluate on
        
    Returns:
        Tuple of (accuracy, metrics_dict)
    """
    evaluator = Evaluator(model, device)
    accuracy, metrics = evaluator.evaluate(test_dataloader)
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    return accuracy, metrics
