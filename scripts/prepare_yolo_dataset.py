"""Convert classification dataset to YOLO format"""
import shutil
from pathlib import Path
import yaml

def convert_to_yolo(classification_dir=None, output_dir=None):
    """Convert classification dataset to YOLO format"""
    base_dir = Path(__file__).parent.parent
    if classification_dir is None:
        classification_dir = base_dir / "garbage-big"
    if output_dir is None:
        output_dir = base_dir / "garbage-detection"
    classification_dir = Path(classification_dir)
    output_dir = Path(output_dir)
    for split in ['train', 'val']:
        src = Path(classification_dir) / split
        if not src.exists():
            print(f"Skipping {split} - not found")
            continue
            
        dst_img = Path(output_dir) / split / 'images'
        dst_lbl = Path(output_dir) / split / 'labels'
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        
        classes = sorted([d.name for d in src.iterdir() if d.is_dir()])
        for cls_id, cls_name in enumerate(classes):
            for img in (src / cls_name).glob('*.jpg'):
                shutil.copy2(img, dst_img / f"{cls_name}_{img.name}")
                with open(dst_lbl / f"{cls_name}_{img.stem}.txt", 'w') as f:
                    f.write(f"{cls_id} 0.5 0.5 1.0 1.0\n")
        
        with open(Path(output_dir) / 'dataset.yaml', 'w') as f:
            yaml.dump({'path': str(Path(output_dir).absolute()), 'train': 'train/images',
                      'val': 'val/images', 'nc': len(classes), 'names': classes}, f)
    print(f"Converted dataset. Classes: {classes}")

if __name__ == '__main__':
    convert_to_yolo()
