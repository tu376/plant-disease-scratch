# Plant Disease Detection using Deep Learning (PlantVillage Dataset)

This project implements a deep learning-based model for plant disease detection using the PlantVillage dataset. The goal is to classify different diseases on plant leaves from images, leveraging a Convolutional Neural Network (CNN) built **from scratch** with CuPy (GPU‑accelerated NumPy) and no high‑level autograd.

## Table of Contents
- [Plant Disease Detection using Deep Learning (PlantVillage Dataset)](#plant-disease-detection-using-deep-learning-plantvillage-dataset)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Dataset](#dataset)
    - [Dataset Structure](#dataset-structure)
  - [Usage](#usage)
    - [Preparing the Dataset](#preparing-the-dataset)
    - [Training the Model](#training-the-model)
    - [Evaluating the Model](#evaluating-the-model)
    - [Making Predictions](#making-predictions)
  - [Model Architecture](#model-architecture)

## Overview

Plant disease detection is a critical task in agriculture. This project builds a **fully custom CNN** from scratch using only CuPy (for GPU acceleration) and Python numerical libraries. No automatic differentiation, no pre‑built layers – everything from convolution to backpropagation is implemented manually.

The model is trained on a subset of the PlantVillage dataset containing **10 disease/healthy categories**. The approach focuses on:

- Manual implementation of Conv2D, MaxPool2D, Linear, ReLU, and Softmax
- Forward and backward passes written from first principles (im2col, col2im, gradient checks)
- Training loop with SGD + momentum
- GPU acceleration via CuPy (falls back to NumPy if GPU unavailable)

## Dataset

This project uses a subset of the [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset). The full dataset contains 54,305 images across 38 classes. For this project, we select **10 representative classes** (e.g., Apple scab, Apple healthy, Tomato Yellow Leaf Curl, etc.) to keep training fast and feasible on a single GPU.

### Dataset Structure
After running the preparation script, the data should be organised as:
data/  
train/  
apple_scab/  
img1.jpg  
img2.jpg  
...  
apple_healthy/  
...  
tomato_curl/  
...  
... (10 classes total)  
val/  
(same class subdirectories, 20% of images)


```text

Images are resized to **64×64 pixels** to reduce computational load while preserving discriminative features.

## Requirements

The following libraries are required:
- Python 3.8 – 3.10
- CuPy (for GPU acceleration) or NumPy (CPU fallback)
- Pillow (PIL)
- Matplotlib
- Scikit-learn (only for confusion matrix)
- tqdm (progress bars)
You can install all dependencies using the provided `requirements.txt`.

## Installation
1. **Clone the repository**:

```bash
git clone https://github.com/your-team/plant-disease-from-scratch.git
cd plant-disease-from-scratch
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate          # Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

**Important:** If you have an NVIDIA GPU with CUDA, install the correct CuPy version:


```bash
# For CUDA 11.x
pip install cupy-cuda11x
# For CUDA 12.x
pip install cupy-cuda12x
```
If you do not have a GPU, the code will automatically fall back to NumPy (slower but still works).

## Usage

### Preparing the Dataset

Run the data preparation script **once** to download the subset, split into train/val, and resize images:

```bash
python prepare_data.py
```
This script:

- Downloads the PlantVillage dataset via `kagglehub` (first time only)
    
- Selects the first 10 classes (configurable)
    
- Splits each class into 80% train / 20% validation
    
- Resizes all images to 64×64 and saves them in the `data/` folder
    

### Training the Model

Start training with default hyperparameters:
```bash
python train.py
```
You can customise training with command‑line arguments:
```bash
python train.py --epochs 30 --batch_size 32 --lr 0.01 --momentum 0.9 --device cuda
```
**Parameters**:

|Argument|Type|Default|Description|
|---|---|---|---|
|`--epochs`|int|30|Number of training epochs|
|`--batch_size`|int|32|Batch size|
|`--lr`|float|0.01|Learning rate|
|`--momentum`|float|0.9|Momentum for SGD|
|`--device`|cuda/cpu|cuda|Device to use (cuda or cpu)|

During training, the best model (lowest validation loss) is saved as `saved_models/best_model.pkl`. Training progress (loss, accuracy) is printed every epoch.

### Evaluating the Model

To evaluate the trained model on the validation set and generate plots:

```bash
python evaluate.py --model saved_models/best_model.pkl
```
This will output:

- Validation accuracy
    
- Confusion matrix (saved as `plots/confusion_matrix.png`)
    
- Loss curves (saved as `plots/loss_curves.png`)
    

### Making Predictions

To classify a single leaf image using a trained model:
```bash
python predict.py --image path/to/leaf.jpg --model saved_models/best_model.pkl
```
Example output:
```text
Top-1 prediction: apple_scab (Confidence: 0.87)
Top-5 predictions:
   1. apple_scab: 87.3%
   2. apple_healthy: 8.1%
   3. tomato_curl: 2.4%
   4. grape_black_rot: 1.2%
   5. corn_common_rust: 1.0%
```
## Model Architecture

The network is a **custom CNN** built entirely from manually implemented layers. It consists of three convolutional blocks followed by a two‑layer fully connected classifier.
```text
Input: (batch, 3, 64, 64)
│
├─ Conv2D(3 → 16, kernel=3, stride=1, padding=1)
├─ ReLU
├─ MaxPool2D(kernel=2, stride=2)            # → (16, 32, 32)
│
├─ Conv2D(16 → 32, kernel=3, stride=1, padding=1)
├─ ReLU
├─ MaxPool2D(kernel=2, stride=2)            # → (32, 16, 16)
│
├─ Conv2D(32 → 64, kernel=3, stride=1, padding=1)
├─ ReLU
├─ MaxPool2D(kernel=2, stride=2)            # → (64, 8, 8)
│
├─ Flatten                                   # → 64 * 8 * 8 = 4096
├─ Linear(4096 → 512)
├─ ReLU
├─ Dropout(0.5) (manually applied during training)
├─ Linear(512 → 10)                          # 10 output classes
└─ Softmax
```
**Total trainable parameters:** ~2.1 million.

All layers (Conv2D, MaxPool2D, Linear, ReLU, Dropout, Softmax) are implemented from scratch using only CuPy/NumPy. Backpropagation for Conv2D is implemented via the **im2col + col2im** trick for efficiency.
