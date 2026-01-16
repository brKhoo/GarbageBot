"""Advanced Training - ResNet50/EfficientNet with early stopping"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
import time
import os
import copy
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Enable optimizations
cudnn.benchmark = True

# Configuration
data_dir = Path(__file__).parent.parent / "garbage-big"
output_dir = Path(__file__).parent.parent / "models"
output_dir.mkdir(exist_ok=True)

# Advanced data augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.33))
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                              shuffle=(x == 'train'), num_workers=8, pin_memory=True)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

print(f"Dataset sizes: {dataset_sizes}")
print(f"Classes: {class_names}")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Use mixed precision training if available
use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler() if use_amp else None


def create_model(model_name='resnet50', pretrained=True):
    """Create model - supports ResNet50, EfficientNet-B3"""
    if model_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(weights='IMAGENET1K_V1' if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def train_model(model, criterion, optimizer, scheduler, num_epochs=50, patience=10):
    """Advanced training function with early stopping and metrics tracking"""
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Forward pass with mixed precision
                with torch.set_grad_enabled(phase == 'train'):
                    if use_amp and phase == 'train':
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    
                    # Backward pass
                    if phase == 'train':
                        if use_amp:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            # Store metrics
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Deep copy the model if validation accuracy improved
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    print(f'\nEarly stopping at epoch {epoch}')
                    print(f'Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}')
                    model.load_state_dict(best_model_wts)
                    time_elapsed = time.time() - since
                    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                    return model, {
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'train_accs': train_accs,
                        'val_accs': val_accs,
                        'best_acc': best_acc.item(),
                        'best_epoch': best_epoch
                    }
    
    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f} at epoch {best_epoch}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_acc': best_acc.item(),
        'best_epoch': best_epoch
    }


def plot_training_history(history, model_name):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['train_losses']) + 1)
    ax1.plot(epochs, history['train_losses'], 'b-', label='Train')
    ax1.plot(epochs, history['val_losses'], 'r-', label='Val')
    ax1.set_title(f'{model_name} - Loss')
    ax1.legend()
    ax2.plot(epochs, history['train_accs'], 'b-', label='Train')
    ax2.plot(epochs, history['val_accs'], 'r-', label='Val')
    ax2.set_title(f'{model_name} - Accuracy')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_training_history.png', dpi=150)
    plt.close()


def main():
    """Train multiple models"""
    models_to_train = [
        ('resnet50', {'lr': 0.001, 'momentum': 0.9}),
        ('efficientnet_b3', {'lr': 0.0005, 'momentum': 0.9}),
        # Add more models as needed
    ]
    
    results = {}
    
    for model_name, hyperparams in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        # Create model (training from scratch, no transfer learning)
        model = create_model(model_name, pretrained=False)
        model = model.to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=hyperparams['lr'], 
                            momentum=hyperparams['momentum'], weight_decay=1e-4)
        
        # Learning rate scheduler with warmup
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        
        # Train
        model, history = train_model(model, criterion, optimizer, scheduler, 
                                    num_epochs=50, patience=10)
        
        # Save model
        model_path = output_dir / f'{model_name}_best.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Save full model for inference
        full_model_path = output_dir / f'{model_name}_full.pth'
        torch.save(model, full_model_path)
        
        # Plot training history
        plot_training_history(history, model_name)
        
        # Store results
        results[model_name] = {
            'best_acc': history['best_acc'],
            'best_epoch': history['best_epoch'],
            'model_path': str(model_path)
        }
    
    # Save results summary
    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("Training Summary:")
    print("="*60)
    for model_name, result in results.items():
        print(f"{model_name}: {result['best_acc']:.4f} accuracy at epoch {result['best_epoch']}")


if __name__ == '__main__':
    main()
