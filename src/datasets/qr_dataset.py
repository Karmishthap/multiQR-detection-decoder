"""QR Code Detection Dataset"""
import os
from pathlib import Path
import yaml
import shutil
from PIL import Image

def prepare_yolo_dataset(train_images_dir, train_labels_dir, output_dir, val_split=0.2):
    """
    Prepare dataset in YOLO format
    Args:
        train_images_dir: path to training images
        train_labels_dir: path to YOLO format labels (.txt files)
        output_dir: output directory for organized dataset
        val_split: validation split ratio
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directory structure
    (output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = sorted(Path(train_images_dir).glob('*'))
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    
    # Split dataset
    split_idx = int(len(image_files) * (1 - val_split))
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Total images: {len(image_files)}")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Copy files
    for img_file in train_files:
        label_file = Path(train_labels_dir) / f"{img_file.stem}.txt"
        if label_file.exists():
            shutil.copy(img_file, output_dir / 'images' / 'train' / img_file.name)
            shutil.copy(label_file, output_dir / 'labels' / 'train' / label_file.name)
    
    for img_file in val_files:
        label_file = Path(train_labels_dir) / f"{img_file.stem}.txt"
        if label_file.exists():
            shutil.copy(img_file, output_dir / 'images' / 'val' / img_file.name)
            shutil.copy(label_file, output_dir / 'labels' / 'val' / label_file.name)
    
    # Create dataset.yaml
    yaml_content = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,  # number of classes (QR code)
        'names': ['qr_code']
    }
    
    with open(output_dir / 'dataset.yaml', 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Dataset prepared at: {output_dir}")
    print(f"Config saved: {output_dir / 'dataset.yaml'}")
    
    return output_dir / 'dataset.yaml'

def validate_dataset(dataset_yaml):
    """Check if dataset is properly formatted"""
    with open(dataset_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    base_path = Path(config['path'])
    train_imgs = base_path / config['train']
    val_imgs = base_path / config['val']
    
    train_count = len(list(train_imgs.glob('*')))
    val_count = len(list(val_imgs.glob('*')))
    
    print(f"✓ Train images: {train_count}")
    print(f"✓ Val images: {val_count}")
    
    return train_count > 0 and val_count > 0