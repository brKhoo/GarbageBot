"""Bluetooth integration - supports both standard and YOLO models"""
import serial
import cv2
import torch
import time
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from classify import visualize_model_predictions

USE_YOLO = False  # Set to True to use YOLO detection

bluetooth = serial.Serial('COM7', 9600)
cap = cv2.VideoCapture(0)
output_dir = "../captured_images"
os.makedirs(output_dir, exist_ok=True)

if USE_YOLO:
    from detect_yolo import GarbageDetector
    detector = GarbageDetector('../garbage_detection_best.pt')
else:
    model_conv = torch.load("../model_conv.pth", weights_only=False)
    model_conv.eval()

def classify_trash():
    ret, frame = cap.read()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_filename = f"{output_dir}/image_{timestamp}.jpg"
    cv2.imwrite(image_filename, frame)
    
    if USE_YOLO:
        primary, conf = detector.get_primary(frame, use_classification=True)
        result = primary if primary else 'trash'
        print(f"Detected: {result} (confidence: {conf:.2f})")
    else:
        result = visualize_model_predictions(model_conv, frame)
        print(f"Detected trash type: {result}")
    
    location = 1
    if result == "biological":
        location = 2
    elif result in ["metal", "plastic"]:
        location = 3
    
    bluetooth.write(f"LOCATION{location}".encode('utf-8'))
    print(f"Sent to location {location}")

try:
    while True:
        if bluetooth.in_waiting > 0:
            if bluetooth.readline().decode('utf-8').strip() == "Shock Detected":
                classify_trash()
except KeyboardInterrupt:
    print("Program interrupted")
finally:
    bluetooth.close()
