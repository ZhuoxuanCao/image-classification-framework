# Torch Image Classification Framework

A modular and extensible image classification training framework built with PyTorch.  
This project supports flexible model configuration, standardized training and evaluation routines, and is suitable for a variety of traditional classification tasks such as object type, color, or surface condition recognition.

## ✅ Features

- **ResNet Support**: Easily switch between ResNet34 and ResNet50 architectures.
- **Custom Dataset Handling**: Supports folder-based dataset organization (`image_train/class_name/image.jpg`).
- **Training Enhancements**: Includes warm-up strategy and cosine annealing learning rate scheduler.
- **Best Model Saving**: Automatically saves the best model checkpoint based on lowest validation loss (with timestamp).
- **TensorBoard Integration**: Visualize training/validation loss and accuracy in real time.
- **Flexible Inference**: Prediction script supports both single image and batch folder inference.
- **Command-Line Configuration**: All major training parameters (batch size, learning rate, model type, etc.) can be controlled via CLI arguments.


