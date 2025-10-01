"""Train YOLOv8 model for QR code detection"""
import argparse
from pathlib import Path
from ultralytics import YOLO
from src.datasets import prepare_yolo_dataset, validate_dataset

def train(args):
    """Train YOLOv8 model"""
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset_yaml = prepare_yolo_dataset(
        train_images_dir=args.train_images,
        train_labels_dir=args.train_labels,
        output_dir=args.data_dir,
        val_split=args.val_split
    )
    
    # Validate dataset
    if not validate_dataset(dataset_yaml):
        raise ValueError("Dataset validation failed!")
    
    # Initialize model
    print(f"\nInitializing {args.model} model...")
    model = YOLO(args.model)
    
    # Train
    print("\nStarting training...")
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
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
    )
    
    # Save best weights path
    weights_dir = Path('weights')
    weights_dir.mkdir(exist_ok=True)
    
    best_model = Path(f'runs/detect/{args.name}/weights/best.pt')
    if best_model.exists():
        import shutil
        shutil.copy(best_model, weights_dir / 'best.pt')
        print(f"\n✓ Best model saved to: weights/best.pt")
    
    print("\n✓ Training complete!")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QR Detection Model")
    
    # Dataset
    parser.add_argument('--train_images', type=str, default='data/train/images',
                        help='Path to training images')
    parser.add_argument('--train_labels', type=str, default='data/train/labels',
                        help='Path to training labels (YOLO format)')
    parser.add_argument('--data_dir', type=str, default='data/prepared',
                        help='Output directory for prepared dataset')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    
    # Model
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt'],
                        help='YOLOv8 model variant')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--device', type=str, default='0',
                        help='Device (0 for GPU, cpu for CPU)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of workers')
    parser.add_argument('--name', type=str, default='qr_detection',
                        help='Experiment name')
    
    args = parser.parse_args()
    train(args)