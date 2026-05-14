"""Main training script for drowsiness detection model."""

import torch
import torch.nn as nn
from src.data import (
    load_dataset_paths,
    create_dataloaders,
    get_train_transform,
    get_eval_transform,
)
from src.model import create_model
from src.training import create_trainer
from src.evaluation import evaluate_model
from src.utils import (
    SEED,
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
)

# Set random seed for reproducibility
torch.manual_seed(SEED)


def main():
    """Main training pipeline."""
    print(f"Using device: {DEVICE}")
    
    # Load dataset
    print("Loading dataset...")
    train_paths, test_paths = load_dataset_paths()
    print(f"Train samples: {len(train_paths)}, Test samples: {len(test_paths)}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_dataloader, test_dataloader = create_dataloaders(
        train_paths,
        test_paths,
        train_transform=get_train_transform(),
        eval_transform=get_eval_transform(),
    )
    
    # Create model
    print("Creating model...")
    model = create_model()
    model = model.to(DEVICE)
    print(model)
    
    # Create trainer
    print("Setting up trainer...")
    trainer = create_trainer(model, learning_rate=LEARNING_RATE, device=DEVICE)
    
    # Train model
    print(f"Training model for {NUM_EPOCHS} epochs...")
    trainer.train(train_dataloader, num_epochs=NUM_EPOCHS)
    
    # Evaluate model
    print("Evaluating model...")
    accuracy, metrics = evaluate_model(model, test_dataloader, device=DEVICE)
    
    # Save model
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    try:
        model_scripted = torch.jit.script(model)
        model_scripted.save(MODEL_SAVE_PATH)
        print(f"Model saved successfully to {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Warning: Could not save as TorchScript: {e}")
        print("Saving as regular PyTorch model...")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    print("\nTraining completed!")
    print(f"Final Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
