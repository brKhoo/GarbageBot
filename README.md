# GarbageBot

GarbageBot is a prototype smart waste-sorting system that uses **machine learning** to classify garbage items and optionally control **hardware (Arduino + motors/servos)** to physically sort them.

The workflow is:
1. Capture an image of an item (e.g., via webcam)
2. Classify the item using a trained ML model
3. Send the predicted category to hardware (optionally over Bluetooth) to sort the item

---

## Features

- Image-based garbage classification using PyTorch
- Webcam capture support for live inference
- Dataset splitting utilities
- Optional Bluetooth communication to Arduino
- Hardware integration for automated sorting

---

## Tech Stack

- **Python** (PyTorch, OpenCV, NumPy)
- **Machine Learning** (CNN / transfer learning)
- **Arduino** (motors / servos for physical sorting)
- **Bluetooth / Serial** communication (optional)

---

## Setup

### Prerequisites
- Python 3.9+
- Webcam (optional)
- Arduino + motors/servos (optional)
- Bluetooth module (optional)

### Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip

pip install torch torchvision opencv-python numpy
```

### Usage
```
python classify.py --image path/to/image.jpg --weights model_ft.pth
```
This classifies an image with the given file path.

```
python webcam.py
```
This captures an image with the currently connected camera or optical device.

```
python split.py
```
This splits a dataset into training/validation/test sets for model training.

```
python bluetooth.py --port /dev/rfcomm0
```
Communicates with an arduino over a connected serial or bluetooth port.

### Hardware Integration
- The Arduino receives class labels from the Python program
- Each label maps to a motor or servo action that physically routes the item
- Communication can be done via USB serial or Bluetooth
- Ensure baud rates and port names match on both the Python and Arduino sides

Typical flow:

- Python classifies the item
- Python sends a label (e.g., plastic, metal, paper)
- Arduino actuates motors/servos based on the label

### Notes
- On Linux, if OpenCV cannot access the webcam, try changing the camera index in webcam.py or checking permissions
- For GPU acceleration, install PyTorch with CUDA support using the official PyTorch installer
- Included model files are ready for inference without retraining
- Bluetooth device names and ports differ across operating systems
