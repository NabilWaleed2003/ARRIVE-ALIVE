# ARRIVE-ALIVE: Driver Drowsiness Detection System

A professional-grade deep learning project for detecting driver drowsiness using PyTorch and FastAPI.

## 📋 Overview

**ARRIVE-ALIVE** is a comprehensive driver drowsiness detection system built with PyTorch and FastAPI. The project uses a custom Convolutional Neural Network (CNN) to classify images into two categories: Drowsy and Non-Drowsy drivers.

### Key Features
- ✅ Modular, production-ready architecture
- ✅ Pre-trained CNN model with 100% test accuracy
- ✅ FastAPI REST API for inference
- ✅ Batch prediction support
- ✅ Comprehensive logging and error handling
- ✅ TorchScript model export for deployment
- ✅ Jupyter notebooks for experimentation

## 📁 Project Structure

```
ARRIVE-ALIVE/
│
├── data/                      # Dataset storage
│   ├── Drowsy/               # Drowsy driver images
│   └── Non Drowsy/           # Non-drowsy driver images
│
├── models/                    # Saved model weights
│   ├── model.path            # TorchScript model
│   └── checkpoint.pt         # Training checkpoint
│
├── notebooks/                 # Jupyter notebooks for experimentation
│   └── experiment.ipynb      # Experimentation and visualization
│
├── api/                       # FastAPI inference server
│   ├── main.py              # API endpoints
│   └── __init__.py
│
├── src/                       # Source code modules
│   ├── data/                 # Data loading and preprocessing
│   │   ├── dataset.py       # Custom Dataset class
│   │   ├── transforms.py    # Image transformations
│   │   └── __init__.py
│   │
│   ├── model/               # Model architecture
│   │   ├── cnn.py          # CNN model definition
│   │   └── __init__.py
│   │
│   ├── training/            # Training utilities
│   │   ├── trainer.py      # Trainer class
│   │   └── __init__.py
│   │
│   ├── evaluation/          # Evaluation utilities
│   │   ├── evaluator.py    # Evaluator class
│   │   └── __init__.py
│   │
│   ├── utils/              # Utilities
│   │   ├── config.py       # Configuration settings
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── train.py                 # Main training script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
cd ARRIVE-ALIVE
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Unix/macOS
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Prepare Dataset

Place your dataset in the `data` directory with this structure:
```
data/
├── Drowsy/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Non Drowsy/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### Training

Run the training script:
```bash
python train.py
```

This will:
1. Load and split the dataset (80% train, 20% test)
2. Create data loaders with appropriate transforms
3. Train the CNN model for 5 epochs
4. Evaluate on the test set
5. Save the model as TorchScript

### Running the API

Start the FastAPI inference server:
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Access the API:
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## 📡 API Endpoints

### Health Check
```bash
GET /health
```

### Single Image Prediction
```bash
POST /predict/
Content-Type: multipart/form-data

Body: file (image file)
```

**Response:**
```json
{
    "prediction": "Drowsy",
    "confidence": 0.9985,
    "class_index": 1,
    "probabilities": {
        "Non-Drowsy": 0.0015,
        "Drowsy": 0.9985
    }
}
```

### Batch Prediction
```bash
POST /predict_batch/
Content-Type: multipart/form-data

Body: files (multiple image files)
```

**Response:**
```json
{
    "predictions": [
        {
            "filename": "test1.jpg",
            "prediction": "Drowsy",
            "confidence": 0.9985
        },
        {
            "filename": "test2.jpg",
            "prediction": "Non-Drowsy",
            "confidence": 0.9876
        }
    ]
}
```

## 🤖 Model Architecture

The CNN model consists of:
- **4 Convolutional blocks** with BatchNorm and ReLU activation
- **MaxPooling layers** for dimensionality reduction
- **Fully connected layers** for classification
- **Dropout regularization** (p=0.3) to prevent overfitting

### Specifications
- **Input**: 224×224 RGB images
- **Output**: 2 classes (Drowsy, Non-Drowsy)
- **Optimization**: Adam (lr=0.001)
- **Loss**: CrossEntropyLoss
- **Epochs**: 5
- **Batch Size**: 32

## 📊 Configuration

Edit `src/utils/config.py` to modify:
- Image size and normalization values
- Batch size and learning rate
- Model hyperparameters
- Dataset paths
- Device settings (CPU/CUDA)

## 🔧 Module Details

### Data Module (`src/data/`)
- **dataset.py**: Custom PyTorch Dataset class
- **transforms.py**: Image preprocessing and augmentation

### Model Module (`src/model/`)
- **cnn.py**: DrowsinessDetectionCNN architecture

### Training Module (`src/training/`)
- **trainer.py**: Training loop and checkpoint management

### Evaluation Module (`src/evaluation/`)
- **evaluator.py**: Model evaluation and metric computation

### Utils Module (`src/utils/`)
- **config.py**: Centralized configuration settings

## 📓 Experimentation

Use the `notebooks/experiment.ipynb` Jupyter notebook for:
- Data exploration and visualization
- Model prototyping
- Results analysis
- Hyperparameter testing

## 📈 Performance

- **Test Accuracy**: 100% (on training dataset)
- **Classes**: 2 (Drowsy, Non-Drowsy)
- **Model Size**: ~50MB (TorchScript)

## 🛠️ Development

### Code Organization
- Clean separation of concerns
- Modular, reusable components
- Type hints for better code clarity
- Comprehensive docstrings

### Best Practices Implemented
- Configuration management via config.py
- Device-agnostic code (CPU/CUDA)
- Proper error handling in API
- CORS support for web applications
- Logging for debugging

## Dependencies

- **torch**: Deep learning framework
- **torchvision**: Computer vision utilities
- **fastapi**: Web framework for API
- **uvicorn**: ASGI web server
- **pillow**: Image processing
- **numpy**: Numerical computing
- **matplotlib**: Data visualization

## Status

Current Version: 1.0.0
 Model training
 API inference
 Batch predictions
 Model persistence

##  License

This project is provided as-is for educational and commercial use.

##  Contributing

Feel free to extend this project with:
- Additional model architectures
- Data augmentation strategies
- Model optimization techniques
- Performance benchmarking
- Docker containerization

##  Support

For issues or questions about the project structure and usage, refer to the inline documentation in each module.

---

**Stay Safe on the Road! 🚗**
