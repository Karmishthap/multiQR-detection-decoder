"""Enhanced training with optimal hyperparameters"""
import argparse
from pathlib import Path
from ultralytics import YOLO
from src.datasets import prepare_yolo_dataset, validate_dataset

def train_enhanced(args):
    """Train with enhanced settings for QR detection"""
    
    print("Preparing dataset...")
    dataset_yaml = prepare_yolo_dataset(
        train_images_dir=args.train_images,
        train_labels_dir=args.train_labels,
        output_dir=args.data_dir,
        val_split=args.val_split
    )
    
    if not validate_dataset(dataset_yaml):
        raise ValueError("Dataset validation failed!")
    
    print(f"\nInitializing {args.model} model...")
    model = YOLO(args.model)
    
    print("\nStarting training with enhanced settings...")
    results = model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        patience=args.patience,
        save=True,
        device=args.device,
        workers=args.workers,
        exist_ok=True,
        
        # Optimized hyperparameters for QR detection
        lr0=0.001,              # Initial learning rate
        lrf=0.01,               # Final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Enhanced augmentation for medicine packs
        hsv_h=0.015,            # Hue
        hsv_s=0.7,              # Saturation
        hsv_v=0.4,              # Value
        degrees=15.0,           # Rotation (increased for tilted QR)
        translate=0.1,          # Translation
        scale=0.5,              # Scale variation
        shear=2.0,              # Shear (helps with perspective)
        perspective=0.0001,     # Perspective transform
        flipud=0.0,             # No vertical flip
        fliplr=0.5,             # Horizontal flip
        mosaic=1.0,             # Mosaic augmentation
        mixup=0.1,              # Mixup augmentation
        copy_paste=0.0,         # Copy-paste augmentation
        
        # Detection settings
        box=7.5,                # Box loss gain
        cls=0.5,                # Class loss gain
        dfl=1.5,                # Distribution focal loss gain
        
        # Multi-scale training
        close_mosaic=10,        # Disable mosaic in final epochs
    )
    
    # Save best weights
    weights_dir = Path('weights')
    weights_dir.mkdir(exist_ok=True)
    
    best_model = Path(f'runs/detect/{args.name}/weights/best.pt')
    last_model = Path(f'runs/detect/{args.name}/weights/last.pt')
    
    if best_model.exists():
        import shutil
        shutil.copy(best_model, weights_dir / 'best.pt')
        print(f"\n✓ Best model saved to: weights/best.pt")
    
    if last_model.exists():
        import shutil
        shutil.copy(last_model, weights_dir / 'last.pt')
        print(f"✓ Last model saved to: weights/last.pt")
    
    print("\n✓ Training complete!")
    print(f"✓ Check results at: runs/detect/{args.name}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced QR Detection Training")
    
    parser.add_argument('--train_images', type=str, default='data/train/images')
    parser.add_argument('--train_labels', type=str, default='data/train/labels')
    parser.add_argument('--data_dir', type=str, default='data/prepared')
    parser.add_argument('--val_split', type=float, default=0.2)
    
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt'],
                        help='Use yolov8s.pt or yolov8m.pt for better accuracy')
    
    parser.add_argument('--epochs', type=int, default=150,
                        help='More epochs for better convergence')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--name', type=str, default='qr_detection_enhanced')
    
    args = parser.parse_args()
    train_enhanced(args)