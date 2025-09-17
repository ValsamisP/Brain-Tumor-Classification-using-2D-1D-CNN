# Brain Tumor Classification using 2D+1D CNN

A deep learnign project for classifying brain tumor types from MRI images using a CNN architecture with transfer learning and attention mechanisms.

## Project Overview

This project implements a brain tumor classification system that achieves 97% test accuracy using a 2D + 1D CNN architecture. The model classifies into four categories:
- Glioma Tumor
- Meningioma Tumor
- No Tumor
- Pituitary Tumor

## Architecture Features
- Transfer Learning: Pre-trained ResNet18 backbone
- Attention Mechanisms: Channel and spatial attention for feature enhancement
- 2D+1D CNN Hybrid: Combines spatial and sequential processing
- Advanced Regularization: Dropout, batch normalization, label smoothing
- Data Augmentation: Augmentation pipeline
- Test Time Augmentation (TTA): Enhanced inference accuracy

## Test Performance

- Test Accuracy -> 96.9%
- Macro F1-Score 97.0%

- Class      Precision      Recall      F1-Score      Support
- Glioma     0.944          0.993       0.968         136
- Meningioma 0.992          0.928       0.959         139
- No Tumor   0.981          0.962       0.971         53
- Pituitary  0.969          0.992       0.981         127

  
<img width="5792" height="2370" alt="image" src="https://github.com/user-attachments/assets/4af46340-6672-40a5-a62d-b9f0acb9b413" />



## Prequisities
Python >= 3.8
CUDA-capable GPU (recommended)

## Clone the Repository
git clone [https://github.com/ValsamisP/brain-tumor-classification.git]
cd brain-tumor-classification

## Install Dependencies
pip install -r requirements.txt

## Basic Training
python main.py --data_dir data_brain_tumor

# Model Architecture

Input (224x224x3) -> ResNet18 Backbone (Pre-trained) -> 2D CNN Enhancement (512→256→128) -> Channel Attention + Spatial Attention -> 1D CNN Processing (128→256→512→256) -> Skip Connections + Classification Head -> Output (4 classes)


## Key Features

- Transfer Learning: ResNet18 pre-trained on ImageNet
- Attention Mechanisms:
  - Channel attention for feature importance
  - Spatial attention for region focus
- 1D CNN Branch: Sequential processing of spatial features
- Skip Connections: Direct feature path to prevent gradient vanishing
- Advanced Regularization: Multiple dropout layers, batch normalization

## Training Features

- Early Stopping: Patience-based training termination
- Learning Rate Scheduling: ReduceLROnPlateau scheduler
- Class Balancing: Weighted sampling for imbalanced datasets


## Test Time Augmentation (TTA)

- Multiple augmented predictions averaging
- Horizontal/vertical flips, noise injection
- Improved robustness and accuracy

## Author
Developed by **Panagiotis Valsamis**, M.Sc. in Data Science candidate and aspiring Data Scientist.

## Education Research Project
This project is developed for educational and research purposes only. It is not intended for clinical use or medical diagnosis

