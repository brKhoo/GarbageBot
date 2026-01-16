# GarbageBot

A robust garbage sorting system using computer vision and machine learning. This project combines image classification and object detection to categorize waste items into 12 different classes, with integration to a physical sorting system via Bluetooth.

## Features

- **Multiple Models**: Both classification (ResNet50/EfficientNet) and object detection (YOLOv8) models
- **Real-time Classification**: Live webcam feed with instant predictions for objects within view
- **Hardware Integration**: Bluetooth/Serial communication with Arduino-based sorting system
- **Training from Scratch**: Models learn entirely from the provided dataset
- **Advanced Training**: Data augmentation, learning rate scheduling, early stopping
- **Extensive Evaluation**: Confusion matrices, ROC curves, per-class metrics

## Technologies Used

- **Python 3.8+**
- **PyTorch** 
- **torchvision**
- **Ultralytics YOLOv8**
- **OpenCV (cv2)**
- **NumPy**
- **scikit-learn**
- **matplotlib**
- **pandas**
- **pyserial**
- **Arduino**

## Project Structure

```
GarbageBot/
├── src/                    # Core functionality
│   ├── classify.py         # Classification module
│   ├── detect_yolo.py     # YOLO object detection module
│   ├── webcam.py          # Real-time webcam classification
│   └── bluetooth.py       # Wireless Arduino integration
│
├── training/               # Model training scripts
│   ├── train_advanced.py  # Classification model training (ResNet50/EfficientNet)
│   └── train_yolo.py      # YOLO object detection training
│
├── evaluation/            # Model evaluation
│   └── evaluate_model.py  # Performance & metrics
│
├── scripts/               # Utility scripts
│   └── prepare_yolo_dataset.py  # Convert dataset to YOLO format
│
├── garbage-big/           # Main training dataset (12 classes, ~10K images)
├── garbage-detection/     # YOLO format dataset (from garbage-big)
├── Arduino/               # Arduino firmware for sorting system
│
├── model_ft.pth           # Fine-tuned classification model
├── model_conv.pth         # Convolutional classification model
└── garbage_detection_best.pt  # YOLO detection model
```

## Get Started

### Requirements

- Python
- Webcam
- Arduino with Bluetooth module (optional, for hardware integration)

### Installation

   ```bash
   # Clone the repo
   git clone https://github.com/brKhoo/GarbageBot
   cd GarbageBot

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install ultralytics --no-deps
   pip install numpy opencv-python pillow pyyaml matplotlib scipy psutil polars requests ultralytics-thop scikit-learn
   ```

### Usage

#### 1. **Real-time Webcam Classification**
   ```bash
   cd src
   python webcam.py
   ```
   - Press 'q' to quit
   - Switch `USE_YOLO = True` in `webcam.py` to use YOLO detection

#### 2. **Hardware Integration (Arduino)**
   ```bash
   cd src
   # Update bluetooth.py to set correct COM port
   python bluetooth.py
   ```
   - Waits for signal from Arduino
   - Captures image, classifies, then sends location command

#### 3. **Train Classification Models**
   ```bash
   cd training
   python train_advanced.py
   ```
   - Trains ResNet50 and EfficientNet-B3 from scratch (no transfer learning)
   - Saves models to `models/` directory
   - Includes data augmentation, early stopping, and LR scheduling

#### 4. **Train YOLO Detection Model**
   ```bash
   # First, convert dataset to YOLO format
   cd scripts
   python prepare_yolo_dataset.py
   
   # Then train YOLO model
   cd ../training
   python train_yolo.py  # Fast training (CPU)
   python train_yolo.py n 50 16 640  # Custom: model_size epochs batch imgsz
   ```

#### 5. **Evaluate Models**
   ```bash
   cd evaluation
   python evaluate_model.py
   ```
   - Generates confusion matrix, ROC curves, per-class metrics
   - Results saved to `evaluation_results/`

## Dataset

The project uses a custom precompiled with **12 garbage categories**:

- `battery`
- `biological`
- `brown-glass`
- `cardboard`
- `clothes`
- `green-glass`
- `metal` 
- `paper`
- `plastic`
- `shoes`
- `trash`
- `white-glass`

**Dataset Structure:**
```
garbage-big/
├── train/
│   ├── battery/
│   ├── biological/
│   └── ...
└── val/
    ├── battery/
    ├── biological/
    └── ...
```

## Models

### Classification Models
- **ResNet50**: Deep residual network (50 layers), trained from scratch
- **EfficientNet-B3**: Efficient architecture with better accuracy/speed tradeoff, trained from scratch
- **Ensemble**: Combines multiple models for improved predictions

### Object Detection Model
- **YOLOv8n**: Nano-sized YOLO model for fast inference
- Trained on full-image bounding boxes (classification mode)

## Configuration

### Webcam Settings
Edit `src/webcam.py`:
```python
USE_YOLO = False  # Set to True for YOLO detection
```

### Bluetooth Settings
Edit `src/bluetooth.py`:
```python
bluetooth = serial.Serial('COM7', 9600)  # Change to your port
USE_YOLO = False  # Set to True for YOLO detection
```

### Training Settings
Edit `training/train_advanced.py`:
```python
model_name = 'resnet50'  # or 'efficientnet_b3'
epochs = 50  # May need more epochs when training from scratch
batch_size = 32
pretrained = False  # Set to True for transfer learning
```

## Evaluation

The evaluation script generates:
- **Confusion Matrix**: Visual representation of classification accuracy
- **Per-Class Metrics**: Precision, Recall, F1-score for each class
- **ROC Curves**: Receiver Operating Characteristic curves
- **Top-K Accuracy**: Accuracy considering top K predictions
- **Classification Report**: Detailed summary