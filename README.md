# Torch Image Classification Framework

A modular and extensible image classification training framework built with PyTorch.  
This project supports flexible model configuration, standardized training and evaluation routines, and is suitable for a variety of traditional classification tasks such as object type, color, or surface condition recognition.

---

## 1. Features

- **ResNet Support**: Easily switch between ResNet34 and ResNet50 architectures.
- **Custom Dataset Handling**: Supports folder-based dataset organization (`image_train/class_name/image.jpg`).
- **Training Enhancements**: Includes warm-up strategy and cosine annealing learning rate scheduler.
- **Best Model Saving**: Automatically saves the best model checkpoint based on lowest validation loss (with timestamp).
- **TensorBoard Integration**: Visualize training/validation loss and accuracy in real time.
- **Flexible Inference**: Prediction script supports both single image and batch folder inference.
- **Command-Line Configuration**: All major training parameters (batch size, learning rate, model type, etc.) can be controlled via CLI arguments.

---

## 2. Project Structure

The project is organized as follows:
```plaintext
image-classification-framework/
├── model/             # Contains ResNet34 and ResNet50 architecture definitions
├── train_img/         # Placeholder for training images (organized by class folders)
│   ├── Class 1               
│   ├── Class 2
│   ├── Class 3               
│   ├── Class 4
│   ├── Class 5                         
├── config.py          # Argument parser and configuration manager
├── dataset.py         # Custom dataset loader and image transforms
├── train.py           # Main training script with warmup, scheduling, and logging
├── predict.py         # Inference script for single image or batch prediction
├── utils.py           # Utility functions, e.g., best model saving with timestamp
├── requirements.txt   # Python dependencies
├── .gitignore         # Files and folders excluded from Git tracking
├── LICENSE            
└── README.md          
```

---

## 3. Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/your_username/image-classification-framework.git
cd image-classification-framework
pip install -r requirements.txt
```
Make sure you are using Python ≥ 3.8 and a CUDA-compatible environment if GPU training is required.

---

## 4. Quick Start

### 4.1 Training

To train a ResNet-based classifier on your custom dataset:

```bash
python train.py \
  --batch_size 32 \
  --epochs 20 \
  --learning_rate 0.001 \
  --model_type resnet34 \
  --dataset ./train_img
```

Training progress will be logged to TensorBoard (`log_dir/`), and the best model (based on validation loss) will be saved in `checkpoints/` with a timestamped filename.

> To monitor training in TensorBoard:
>
> ```bash
> tensorboard --logdir=log_dir
> ```

### 4.2 Inference (Prediction)

To run inference on a single image or an entire folder of images:

```bash
python predict.py \
  --input ./path/to/image_or_folder \
  --model_type resnet34 \
  --save_path ./checkpoints/resnet34_epoch3_val0.0013_lr0.001_bs32_20250601_125517.pth
```

The script will print predicted class names and their confidence scores to the console.

---

## 5. Dataset Structure

The training dataset should follow the folder-based structure accepted by `torchvision.datasets.ImageFolder`, as shown below:

```
train_img/
├── class_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class_2/
│   ├── image3.jpg
│   ├── image4.jpg
│   └── ...
```

* Each subdirectory name (`class_1`, `class_2`, ...) will be treated as the class label.
* All images inside a class folder will be used as training/validation data.

---

## 6. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
