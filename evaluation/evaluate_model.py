"""Model Evaluation - Confusion matrix, metrics, ROC curves"""

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                            precision_recall_fscore_support, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from itertools import cycle
import os
from pathlib import Path
import json

# Configuration
data_dir = Path(__file__).parent.parent / "garbage-big"
model_path = Path(__file__).parent.parent / "model_conv.pth"  # Change to your model
output_dir = Path(__file__).parent.parent / "evaluation_results"
output_dir.mkdir(exist_ok=True)

# Data transforms (same as validation)
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, 
                                         shuffle=False, num_workers=4)
class_names = val_dataset.classes
num_classes = len(class_names)

device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def load_model(model_path, num_classes):
    """Load model from checkpoint"""
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    model = model.to(device)
    return model


def evaluate_model(model, dataloader, class_names, top_k=3):
    """Comprehensive model evaluation"""
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Overall accuracy
    accuracy = np.mean(all_preds == all_labels)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Top-k accuracy
    top_k_acc = calculate_top_k_accuracy(all_probs, all_labels, top_k)
    print(f"Top-{top_k} Accuracy: {top_k_acc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, output_dir / 'confusion_matrix.png')
    
    # Classification report
    report = classification_report(all_labels, all_preds, 
                                   target_names=class_names, 
                                   output_dict=True)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None)
    
    per_class_metrics = {
        class_name: {
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1_score),
            'support': int(supp)
        }
        for class_name, prec, rec, f1_score, supp in 
        zip(class_names, precision, recall, f1, support)
    }
    
    # Plot per-class metrics
    plot_per_class_metrics(per_class_metrics, output_dir / 'per_class_metrics.png')
    
    # ROC curves (one-vs-rest)
    plot_roc_curves(all_probs, all_labels, class_names, 
                   output_dir / 'roc_curves.png')
    
    # Save results
    results = {
        'overall_accuracy': float(accuracy),
        f'top_{top_k}_accuracy': float(top_k_acc),
        'per_class_metrics': per_class_metrics,
        'classification_report': report
    }
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def calculate_top_k_accuracy(probs, labels, k=3):
    """Calculate top-k accuracy"""
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    correct = 0
    for i, label in enumerate(labels):
        if label in top_k_preds[i]:
            correct += 1
    return correct / len(labels)


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_per_class_metrics(metrics, save_path):
    """Plot precision, recall, and F1-score per class"""
    classes = list(metrics.keys())
    precision = [metrics[c]['precision'] for c in classes]
    recall = [metrics[c]['recall'] for c in classes]
    f1 = [metrics[c]['f1_score'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Per-class metrics plot saved to {save_path}")


def plot_roc_curves(probs, labels, class_names, save_path):
    """Plot ROC curves for each class"""
    labels_bin = label_binarize(labels, classes=range(len(class_names)))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(labels_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-avg (AUC={roc_auc["micro"]:.2f})', 
             color='deeppink', linestyle=':', linewidth=2)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple'])
    for i, color in zip(range(min(6, len(class_names))), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{class_names[i]} (AUC={roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"ROC curves saved to {save_path}")




if __name__ == '__main__':
    print("Loading model...")
    model = load_model(model_path, num_classes)
    
    print("Evaluating model...")
    results = evaluate_model(model, val_loader, class_names)
    
    print(f"\nEvaluation complete! Results saved to {output_dir}/")
