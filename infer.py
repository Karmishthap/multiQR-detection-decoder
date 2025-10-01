"""Inference script for QR code detection"""
import argparse
from pathlib import Path
from ultralytics import YOLO
from src.utils import get_image_files, create_detection_output, save_json
from tqdm import tqdm

def infer(args):
    """Run inference on images and save results"""
    
    # Load model
    print(f"Loading model from: {args.weights}")
    model = YOLO(args.weights)
    
    # Get image files
    image_files = get_image_files(args.input)
    print(f"Found {len(image_files)} images")
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {args.input}")
    
    # Run inference
    results_list = []
    
    for img_path in tqdm(image_files, desc="Processing images"):
        # Get image ID (filename without extension)
        image_id = img_path.stem
        
        # Predict
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False
        )
        
        # Extract bounding boxes
        bboxes = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # [x_min, y_min, x_max, y_max]
            
            for box in boxes:
                x_min, y_min, x_max, y_max = map(float, box)
                bboxes.append([x_min, y_min, x_max, y_max])
        
        # Create output entry
        output_entry = create_detection_output(image_id, bboxes)
        results_list.append(output_entry)
    
    # Save results
    save_json(results_list, args.output)
    print(f"\n✓ Results saved to: {args.output}")
    print(f"✓ Total images processed: {len(results_list)}")
    
    # Statistics
    total_qr = sum(len(r['qrs']) for r in results_list)
    avg_qr = total_qr / len(results_list) if results_list else 0
    print(f"✓ Total QR codes detected: {total_qr}")
    print(f"✓ Average QR per image: {avg_qr:.2f}")
    
    return results_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QR Code Detection Inference")
    
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input images directory')
    parser.add_argument('--output', type=str, default='outputs/submission_detection_1.json',
                        help='Output JSON file path')
    parser.add_argument('--weights', type=str, default='weights/best.pt',
                        help='Path to trained model weights')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for inference')
    parser.add_argument('--device', type=str, default='0',
                        help='Device (0 for GPU, cpu for CPU)')
    
    args = parser.parse_args()
    infer(args)