"""Webcam classification - supports both standard and YOLO models"""
import cv2
import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from classify import visualize_model_predictions

USE_YOLO = False  # Set to True to use YOLO detection

if USE_YOLO:
    from detect_yolo import GarbageDetector
    detector = GarbageDetector('../garbage_detection_best.pt', conf=0.25)
else:
    model_ft = torch.load("../model_ft.pth", weights_only=False, map_location=torch.device('cpu'))
    model_ft.eval()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cv2.namedWindow("Classification", cv2.WINDOW_NORMAL)

print("Classification Active! Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if USE_YOLO:
        primary, conf = detector.get_primary(frame, use_classification=True)
        text = f"{primary} ({conf:.2f})" if primary else "No detection"
    else:
        text = visualize_model_predictions(model_ft, frame)
    
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Classification", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
