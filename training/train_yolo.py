"""Train YOLOv8 - Fast Training Configuration"""
from ultralytics import YOLO
from pathlib import Path
import shutil

def train_fast(model_size='n', epochs=30, batch=8, imgsz=416):
    """
    Fast training configuration for CPU/GPU
    - Smaller model (nano)
    - Fewer epochs (30 with early stopping)
    - Smaller images (416px = 2.4x faster than 640px)
    - Smaller batch (8 for CPU, increase for GPU)
    """
    base_dir = Path(__file__).parent.parent
    dataset = base_dir / "garbage-detection" / "dataset.yaml"
    if not dataset.exists():
        print(f"Error: {dataset} not found!")
        print("Run: python scripts/prepare_yolo_dataset.py first")
        return
    
    model = YOLO(f'yolov8{model_size}.pt')
    results = model.train(
        data=str(dataset), epochs=epochs, batch=batch, imgsz=imgsz,
        device='cpu',  # Change to 0 for GPU
        project=str(base_dir / 'detection_results'), name='garbage_detection',
        patience=10, save_period=5, plots=False, verbose=True
    )
    
    best = base_dir / 'detection_results' / 'garbage_detection' / 'weights' / 'best.pt'
    if best.exists():
        shutil.copy2(best, base_dir / 'garbage_detection_best.pt')
        print(f"Model saved to {base_dir / 'garbage_detection_best.pt'}")

if __name__ == '__main__':
    import sys
    # Usage: python train_yolo.py [model_size] [epochs] [batch] [imgsz]
    # Fast: python train_yolo.py n 30 8 416  (~6-8 hours CPU, ~20 min GPU)
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    train_fast(
        model_size=args[0] if len(args) > 0 else 'n',
        epochs=int(args[1]) if len(args) > 1 else 30,
        batch=int(args[2]) if len(args) > 2 else 8,
        imgsz=int(args[3]) if len(args) > 3 else 416
    )
