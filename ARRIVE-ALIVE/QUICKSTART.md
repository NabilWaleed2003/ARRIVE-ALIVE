# 🚀 ARRIVE-ALIVE Quick Start Guide

Welcome to the Driver Drowsiness Detection project!

## Quick Start (5 minutes)

### 1️⃣ Setup

```bash
cd ARRIVE-ALIVE
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ Prepare Dataset

Place images in:
```
Driver Drowsiness Dataset (DDD)/
├── Drowsy/
└── Non Drowsy/
```

### 3️⃣ Train Model

```bash
python train.py
```

### 4️⃣ Start API

```bash
python -m uvicorn api.main:app --reload
```

Visit: **http://localhost:8000/docs**

### 5️⃣ Test Image

```bash
python inference.py path/to/image.jpg
```

## 📁 Project Structure

```
├── src/           - Modular code (data, model, training, evaluation, utils)
├── api/           - FastAPI server
├── models/        - Saved models
├── data/          - Your dataset
├── notebooks/     - Jupyter notebooks
├── train.py       - Training script
├── inference.py   - Testing script
└── requirements.txt
```

## 📚 Documentation

- See **README.md** for full documentation
- Check **src/** module docstrings for details
- Review **src/utils/config.py** to customize parameters

## 🎯 Main Commands

| Command | Purpose |
|---------|---------|
| `python train.py` | Train the model |
| `python test_setup.py` | Verify installation |
| `python -m uvicorn api.main:app --reload` | Start API |
| `python inference.py <image>` | Test prediction |
| `jupyter notebook notebooks/experiment.ipynb` | Exploration |

## 🔧 Configuration

Edit `src/utils/config.py` to change:
- Image size, batch size, learning rate
- Number of epochs
- Model hyperparameters
- Data paths

---

**Happy coding! 🚀**
