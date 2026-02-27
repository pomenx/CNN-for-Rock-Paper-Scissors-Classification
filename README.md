# CNN for Rock-Paper-Scissors Classification

A deep learning project that uses Convolutional Neural Networks (CNN) to classify images of Rock, Paper, and Scissors gestures.

## 📋 Description

This project implements and trains multiple CNN architectures to recognize and classify the three basic Rock-Paper-Scissors game gestures from images. The project includes:

- **Multiple network architectures** (Net, DeepNet, NetDropout)
- **Advanced data augmentation** including background replacement
- **Complete evaluation system** with classification metrics
- **GPU support** for fast training

## 🚀 Setup and Installation

### Prerequisites
- Python 3.8+
- PyTorch with CUDA support (optional, for speed)
- Conda or pip

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pomenx/CNN-for-Rock-Paper-Scissors-Classification.git
   cd CNN-for-Rock-Paper-Scissors-Classification
   ```

2. **Create a Python environment (recommended):**
   ```bash
   conda create -n cnn-rps python=3.9
   conda activate cnn-rps
   ```

3. **Install dependencies:**
   ```bash
   pip install requirements.txt
   ```

## 📁 Project Structure

```
CNN-for-Rock-Paper-Scissors-Classification/
├── CNN.py                 # Network architectures definition
├── train.py              # Training and validation script
├── rps_dataloader.py     # Data loader with augmentation
├── README.md             # This file
└── dataset/
    ├── train/
    │   ├── rock/         # Training images of rock gesture
    │   ├── paper/        # Training images of paper gesture
    │   └── scissors/     # Training images of scissors gesture
    └── test/
        ├── rock/         # Test images of rock gesture
        ├── paper/        # Test images of paper gesture
        └── scissors/     # Test images of scissors gesture
```

## 🤖 Available Architectures

### 1. BaselineNet
Lightweight architecture with 2 convolutional layers:
- 2 convolutional layers (3→8→4 channels)
- Max pooling after each convolution
- Tanh activation
- 2 fully connected layers

### 2. DeepNet
Deeper architecture with 3 convolutional layers:
- 3 convolutional layers (3→8→8→4 channels)
- Max pooling after each convolution
- ReLU activation
- 2 fully connected layers

### 3. NetDropout
DeepNet version with dropout to reduce overfitting
