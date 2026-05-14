# EO-SAR Change Detection using Siamese U-Net

## Overview

This project implements a deep learning pipeline for EO-SAR change detection using a Siamese U-Net architecture. The model detects building damage and structural changes between pre-event and post-event satellite imagery.

The solution was developed as part of the AI Research Intern technical assessment for GalaxEye Space.

---

# Problem Statement

The objective of this task is to identify changes between:
- pre-event satellite imagery
- post-event satellite imagery

The model predicts binary change masks indicating damaged or changed regions.

---

# Dataset Structure

``` id="4f0n1a"
data/
│
├── train/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
│
├── val/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
│
└── test/
    ├── pre-event/
    ├── post-event/
    └── target/





# Approach
Model Architecture

A Siamese U-Net based segmentation architecture was used.

Pipeline:
Load pre-event and post-event TIFF images
Normalize imagery
Concatenate image channels
Train segmentation network
Predict binary change masks
Technologies Used
Python
PyTorch
OpenCV
NumPy
TIFFFile
CUDA
tqdm
Installation
Create Virtual Environment
python -m venv venv
Activate Environment
Windows
venv\Scripts\activate
Install Dependencies
pip install -r requirements.txt
CUDA Installation

GPU-enabled PyTorch was installed using:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
Training

Run training using:

python train.py

The model automatically saves the best checkpoint as:

best_model.pth
Evaluation

Run evaluation using:

python eval.py
Final Test Results
Metric	Score
Precision	0.0052
Recall	0.1557
IoU	0.0050
F1 Score	0.0093
Challenges Faced
EO and SAR modality differences
Highly imbalanced change masks
Limited GPU memory (2GB MX450)
TIFF image preprocessing
Large dataset size
Future Improvements
Attention U-Net
Transformer-based architectures
Better loss balancing
Data augmentation
Class imbalance handling
Multi-scale feature fusion
Project Structure
Task/
│
├── data/
├── models/
│   └── unet_siamese.py
│
├── utils/
│   ├── dataset.py
│   └── metrics.py
│
├── train.py
├── eval.py
├── requirements.txt
├── README.md
└── best_model.pth
Hardware Used
NVIDIA GeForce MX450 GPU
CUDA-enabled PyTorch
Windows 11
Author

Biswajeet Dixit

AI Engineer | Generative AI | Computer Vision | Deep Learning