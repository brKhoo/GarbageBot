"""Unified Classification Module - Single model and ensemble predictions"""
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms, datasets
from PIL import Image
import os
from pathlib import Path

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = Path(__file__).parent.parent / "garbage-big"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms)
                  for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else "cpu")


def visualize_model_predictions(model, frame):
    """Basic classification - returns class name"""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = data_transforms(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        return class_names[preds[0]]


def classify_with_confidence(model, frame, class_names=class_names, device=device):
    """Single model classification with confidence score"""
    model.eval()
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = data_transforms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)
        return class_names[pred_idx.item()], confidence.item()


class EnsembleClassifier:
    """Ensemble classifier using multiple models"""
    def __init__(self, model_paths, class_names=class_names, device=device):
        self.models = []
        self.class_names = class_names
        self.device = device
        for model_path in model_paths:
            if os.path.exists(model_path):
                model = torch.load(model_path, map_location=device, weights_only=False)
                model.eval().to(device)
                self.models.append(model)
        if len(self.models) == 0:
            raise ValueError("No valid models loaded!")
        print(f"Loaded {len(self.models)} models for ensemble")
    
    def predict(self, frame, return_confidence=True):
        """Make ensemble prediction"""
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = data_transforms(img).unsqueeze(0).to(self.device)
        all_probs = []
        with torch.no_grad():
            for model in self.models:
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        avg_probs = np.mean(all_probs, axis=0)[0]
        pred_idx = np.argmax(avg_probs)
        confidence = avg_probs[pred_idx]
        predicted_class = self.class_names[pred_idx]
        return (predicted_class, confidence) if return_confidence else predicted_class
    
    def get_primary(self, frame):
        """Get most confident prediction"""
        return self.predict(frame, return_confidence=True)
