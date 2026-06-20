# Plant Leaf Disease Classification using a CNN Built from Scratch

This project implements an end-to-end plant leaf disease classification system using a custom Convolutional Neural Network (CNN) developed entirely from scratch with **CuPy**. No automatic differentiation framework (PyTorch, TensorFlow, Keras, etc.) is used for the CNN implementation. All layers, forward passes, backward passes, and parameter updates are manually implemented.

The project uses images from the PlantVillage dataset and evaluates both:

* A direct CNN classifier
* Classical machine learning classifiers trained on CNN-extracted features:

  * Support Vector Machine (SVM)
  * XGBoost
  * Random Forest
  * k-Nearest Neighbours (k-NN)
  * Logistic Regression

---

# Table of Contents

* Overview
* Dataset
* Project Structure
* Installation
* Usage

  * Dataset Preparation
  * CNN Training
  * Feature-Based Machine Learning Models
* CNN Architecture
* Experimental Configurations
* Results

---

# Overview

Plant diseases are a major threat to agricultural productivity. Early detection enables timely treatment and reduces crop losses.

This project investigates whether a compact CNN implemented entirely from first principles can learn meaningful visual representations for plant disease recognition.

Key features:

* CNN implemented from scratch using CuPy
* Manual Conv2D, MaxPool2D, Linear, ReLU, Flatten layers
* Manual backpropagation
* He (Kaiming) weight initialization
* Cross-Entropy Loss and Focal Loss
* SGD with Momentum and Adam optimizers
* CNN feature extraction for downstream classifiers
* GPU acceleration through CuPy

---

# Dataset

Dataset source:

PlantVillage Dataset

The original dataset contains more than 50,000 plant leaf images across multiple crops and disease categories.

For this project:

* The first 20 classes (alphabetically sorted) are selected
* Images are resized to 64×64 RGB
* Dataset is split before augmentation:

  * 80% Training
  * 10% Validation
  * 10% Testing
* Data augmentation is applied only to the training set
* Each class is balanced to 500 training images

### Augmentation Operations

For underrepresented classes, one of the following transformations is randomly applied:

* Random rotation (-20° to +20°)
* Horizontal flip
* Random brightness adjustment
* Random contrast adjustment

Validation and test sets contain only original images to prevent data leakage.

---

# Project Structure

```text
project/
│
├── preprocessing.py
├── train_cnn.py
├── train_ml.py
├── evaluate.py
│
├── layers/
│   ├── conv2d.py
│   ├── maxpool.py
│   ├── linear.py
│   ├── relu.py
│   └── flatten.py
│
├── model/
│   ├── cnn.py
│   ├── losses.py
│   ├── optimizers.py
│   └── ...
│
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── saved_models/
├── features/
└── README.md
```

# Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\Scripts\activate
```

Linux / Mac:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For NVIDIA GPUs:

CUDA 11.x

```bash
pip install cupy-cuda11x
```

CUDA 12.x

```bash
pip install cupy-cuda12x
```

---

# Usage

## Dataset Preparation

Run:

```bash
python preprocessing.py
```

This script:

* Downloads PlantVillage
* Selects the first 20 classes
* Splits train/validation/test
* Applies augmentation
* Resizes images to 64×64
* Creates balanced training data

---

## CNN Training

Default training:

```bash
python train_cnn.py
```

Example:

```bash
python train_cnn.py \
    --epochs 10 \
    --batch_size 4 \
    --lr 0.001 \
    --optimizer adam \
    --loss cross_entropy
```

Available options:

Optimizer:

* adam
* sgd

Loss:

* cross_entropy
* focal

The best validation checkpoint is automatically saved.

---

## Training Classical ML Models

After CNN training, extract CNN features and train a machine learning classifier:

```bash
python train_ml.py --model svm
```

Available models:

```text
svm
xgboost
random_forest
knn
logistic_regression
```

Examples:

```bash
python train_ml.py --model svm
python train_ml.py --model xgboost
python train_ml.py --model random_forest
python train_ml.py --model knn
python train_ml.py --model logistic_regression
```

---

# CNN Architecture

Input:

```text
3 × 64 × 64 RGB Image
```

Architecture:

```text
Conv2D(3 → 16, kernel=3, padding=1)
ReLU
MaxPool2D(2×2)

Conv2D(16 → 32, kernel=3, padding=1)
ReLU
MaxPool2D(2×2)

Flatten

Linear(8192 → 128)
ReLU

Linear(128 → 20)
```

Feature extraction:

The 128-dimensional activation vector from the penultimate layer is used as input for all downstream machine learning models.

---

# Experimental Configurations

The CNN is evaluated under four configurations:

| Loss Function | Optimizer      |
| ------------- | -------------- |
| Cross Entropy | SGD + Momentum |
| Cross Entropy | Adam           |
| Focal Loss    | SGD + Momentum |
| Focal Loss    | Adam           |

---

# Results

| Configuration        | CNN    | XGBoost | SVM    | Random Forest | k-NN   | Logistic Regression |
| -------------------- | ------ | ------- | ------ | ------------- | ------ | ------------------- |
| CE + SGD Momentum    | 92.46% | 90.32%  | 93.40% | 92.25%        | 92.37% | 92.72%              |
| CE + Adam            | 89.80% | 90.23%  | 93.66% | 91.37%        | 92.20% | 91.82%              |
| Focal + SGD Momentum | 89.55% | 89.29%  | 92.72% | 91.46%        | 89.72% | 90.57%              |
| Focal + Adam         | 88.86% | 88.77%  | 91.60% | 89.72%        | 90.75% | 90.40%              |

Best overall result:

* SVM on CNN features
* Accuracy: 93.66%

---

# Technologies Used

* Python
* CuPy
* NumPy
* Pillow
* Scikit-learn
* XGBoost
* Matplotlib
* Joblib

---

# License

This project is intended for educational and research purposes.
