"""Configuration settings for the Driver Drowsiness Detection project."""

import torch

# Dataset paths
DATA_DIR = "Driver Drowsiness Dataset (DDD)"
DROWSY_DIR = "Drowsy"
NON_DROWSY_DIR = "Non Drowsy"

# Image settings
IMAGE_SIZE = 224
CHANNELS = 3

# Normalization values (calculated from training data)
MEAN = [0.5013777017593384, 0.381178617477417, 0.33691293001174927]
STD = [0.2515166103839874, 0.22795365750789642, 0.21324042975902557]

# Training settings
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
TRAIN_TEST_SPLIT = 0.8

# Model settings
NUM_CLASSES = 2
DROPOUT_RATE = 0.3

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model paths
MODEL_SAVE_PATH = "models/model.path"
MODEL_CHECKPOINT_PATH = "models/checkpoint.pt"

# Random seed
SEED = 42

# Class labels
CLASS_NAMES = {0: "Non-Drowsy", 1: "Drowsy"}
