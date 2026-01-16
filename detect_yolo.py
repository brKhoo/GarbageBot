"""YOLOv8 Object Detection - Concise Implementation"""
import cv2
from pathlib import Path
from ultralytics import YOLO
import torch

class GarbageDetector:
    def __init__(self, model_path='garbage_detection_best.pt', conf=0.25):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to trained model
            conf: Confidence threshold (default 0.25, lower = more detections)
        """
        self.conf = conf
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if Path(model_path).exists():
            self.model = YOLO(model_path)
            print(f"Loaded model: {model_path}")
        else:
            print(f"Warning: {model_path} not found, using pretrained YOLOv8n")
            self.model = YOLO('yolov8n.pt')
    
    def detect(self, frame, return_image=False, filter_full_image=True):
        """
        Detect objects: returns list of {class, confidence, bbox} or (detections, annotated_frame)
        
        Args:
            filter_full_image: If True, filters out detections covering >85% of image (full-image boxes)
        """
        results = self.model(frame, conf=self.conf)
        detections = []
        h, w = frame.shape[:2]
        frame_area = h * w
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # Calculate bounding box area
                bbox_area = (x2 - x1) * (y2 - y1)
                coverage = bbox_area / frame_area
                
                # Filter out full-image detections (covers >85% of frame)
                if filter_full_image and coverage > 0.85:
                    continue
                
                detections.append({
                    'class': r.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': [x1, y1, x2, y2]
                })
        
        if return_image:
            img = frame.copy()
            for d in detections:
                x1, y1, x2, y2 = d['bbox']
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{d['class']}: {d['confidence']:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return detections, img
        return detections
    
    def get_primary(self, frame, use_classification=True):
        """
        Get primary detection: (class_name, confidence)
        
        Args:
            use_classification: If True, uses whole-image classification (better for this model)
                               If False, uses object detection (may be empty due to filtering)
        """
        if use_classification:
            # For models trained with full-image boxes, treat as classification
            # Use very low confidence to ensure we get a result
            results = self.model(frame, conf=0.01)  # Very low threshold to get any detection
            if results and len(results[0].boxes) > 0:
                # Get the most confident detection
                boxes = results[0].boxes
                best_idx = boxes.conf.argmax().item()
                box = boxes[best_idx]
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = results[0].names[cls_id]
                return class_name, conf
            return None, 0.0
        else:
            # Use filtered detections
            dets = self.detect(frame, filter_full_image=True)
            return (max(dets, key=lambda x: x['confidence'])['class'], 
                    max(dets, key=lambda x: x['confidence'])['confidence']) if dets else (None, 0.0)
