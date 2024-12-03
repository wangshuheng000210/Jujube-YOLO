Here is the updated README with the `epochs` changed to 200:

---

# Jujube-YOLO 🍏

**Jujube-YOLO** is an object detection model based on the YOLOv11 architecture, specifically designed for jujube fruit detection and classification tasks. This model integrates your original **DCSE (Double Convolution Squeeze-and-Excitation)** and **MBCA (Multi-Branch Channel Attention)** modules, optimizing it for agricultural environments and achieving outstanding precision and efficiency in object detection.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Training](#training)
- [Usage](#usage)

## Overview

**Jujube-YOLO** is a YOLOv11-based model tailored for jujube fruit detection tasks. The model is customized with two original modules: **DCSE** and **MBCA**, which enhance feature extraction, channel attention, and multi-scale feature fusion. It excels in detecting and classifying different jujube fruit types, even in complex agricultural environments, with an emphasis on high accuracy and efficiency.

### Key Features:
- **Jujube Classification**: Accurately detects and classifies “Good” and “Cracked” jujube fruits.
- **Original Modules**: Integrates **DCSE** and **MBCA** for enhanced feature extraction and attention mechanisms.
- **Multi-Scale Detection**: Optimized for detecting objects of various scales, especially suitable for agricultural object detection tasks.
- **Efficient**: Improved computational efficiency through custom modules, suitable for real-time applications.

## Features

- **High Detection Accuracy**: Particularly effective in distinguishing between different types of jujube fruits (Good and Cracked).
- **Multi-Scale Processing**: The model can handle objects of varying sizes through multi-branch structures and custom modules.
- **Original Modules**:
  - **DCSE**: The Double Convolution Squeeze-and-Excitation module improves channel interaction and enhances both local and global feature representation.
  - **MBCA**: The Multi-Branch Channel Attention module focuses on extracting important features while suppressing redundant information.
- **Efficient Computation**: Compared to traditional convolution layers, the model improves computational efficiency while maintaining high accuracy.
- **Custom Backbone Network**: The architecture is designed to optimize feature extraction and detection.

## Model Architecture

**Jujube-YOLO** is based on the YOLOv11 architecture and customized with the following core modules:

### Backbone:
1. **DCSE (Double Convolution Squeeze-and-Excitation Module)**: Enhances feature extraction and channel interaction.
2. **C3k2_RCM (Rectangular Self-Calibrated Module)**: Uses self-calibration and multi-scale feature extraction for better performance in complex environments.
3. **SPPF**: Spatial Pyramid Pooling Fusion module for improved multi-scale perception.
4. **C2PSA**: Channel Adaptive Pooling Module for further multi-scale feature fusion.

### Head:
1. **Upsample**: Upsampling module to restore spatial resolution.
2. **Concat**: Concatenates feature maps from different layers for better feature fusion.
3. **MBCA (Multi-Branch Channel Attention Module)**: Enhances key feature extraction, improving accuracy.
4. **Detect**: Outputs final detection results, including bounding boxes and fruit categories.

### Output:
- **Detection Results**: Bounding boxes and class labels for each detected jujube fruit (Good or Cracked).

## Installation

To install and use **Jujube-YOLO**, you'll need Python 3.8+ and the required dependencies.

### Dependencies:
- Python 3.8+
- PyTorch 1.9.0+
- OpenCV
- numpy
- Matplotlib
- CUDA (if using GPU acceleration)

### Installation Steps:

 Clone the repository:
    ```bash
    git clone https://github.com/wangshuheng000210/Jujube-YOLO.git
    cd jujube-yolo
  

## Training

To train **Jujube-YOLO** on your custom dataset, follow these steps:

1. **Prepare the Dataset**: Ensure your dataset is annotated in YOLO format with the correct labels (Good and Cracked jujube fruits).
   
2. **Modify the Training YAML Configuration**: Edit the configuration file to include your dataset paths and class labels.

3. **Train the Model**:
    ```bash
    python train.py --img 640 --batch 16 --epochs 200 --data jujube_data.yaml --weights '' --device 0
    ```
    - `--img 640`: Input image size.
    - `--batch 16`: Batch size.
    - `--epochs 200`: Number of training epochs.
    - `--data jujube_data.yaml`: Path to the dataset configuration file.
    - `--weights ''`: Start training from scratch (or provide pre-trained model path).

## Usage

Once training is complete, you can use the trained model for inference.


This README provides an overview, installation, training, and usage instructions for **Jujube-YOLO**, highlighting the original modules **DCSE** and **MBCA** and their impact on improving detection performance in agricultural environments.
