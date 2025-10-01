"""Enhanced inference with Test-Time Augmentation"""
import argparse
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO
from src.utils import get_image_files, create_detection_output, save_json, nms
from tqdm import tqdm

def rotate_bbox(bbox, angle, img_shape):
    """Rotate bbox back to original orientation"""
    h, w = img_shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Center of image
    cx, cy = w / 2, h / 2
    
    # Bbox corners
    corners = np.array([
        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
    ])
    
    # Rotation matrix
    angle_rad = np.radians(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    # Rotate around center
    rotated = []
    for x, y in corners:
        x_new = cos_a * (x - cx) + sin_a * (y - cy) + cx
        y_new = -sin_a * (x - cx) + cos_a * (y - cy) + cy
        rotated.append([x_new, y_new])
    
    rotated = np.array(rotated)
    x_min, y_min = rotated.min(axis=0)
    x_max, y_max = rotated.max(axis=0)
    
    return [max(0, x_min), max(0, y_min), min(w, x_max), min(h, y_max)]

def detect_with_tta(model, image, conf=0.15, iou=0.45, imgsz=640):
    """Detect with Test-Time Augmentation"""
    all_boxes = []
    all_scores = []
    
    # Original
    results = model.predict(image, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        all_boxes.extend(boxes)
        all_scores.extend(scores)
    
    # Horizontal flip
    flipped = cv2.flip(image, 1)
    results = model.predict(flipped, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        # Flip boxes back
        w = image.shape[1]
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            all_boxes.append([w - x2, y1, w - x1, y2])
            all_scores.append(score)
    
    # Multi-scale
    for scale in [0.8, 1.2]:
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(image, (new_w, new_h))
        
        results = model.predict(scaled, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            # Scale boxes back
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box
                all_boxes.append([x1/scale, y1/scale, x2/scale, y2/scale])
                all_scores.append(score * 0.9)  # Slightly lower confidence
    
    # Apply NMS to remove duplicates
    if len(all_boxes) > 0:
        keep_idx = nms(all_boxes, all_scores, iou_threshold=0.5)
        final_boxes = [all_boxes[i] for i in keep_idx]
        return final_boxes
    
    return []

def infer_enhanced(args):
    """Enhanced inference with TTA"""
    
    print(f"Loading model from: {args.weights}")
    model = YOLO(args.weights)
    
    image_files = get_image_files(args.input)
    print(f"Found {len(image_files)} images")
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {args.input}")
    
    results_list = []
    
    for img_path in tqdm(image_files, desc="Processing images"):
        image_id = img_path.stem
        image = cv2.imread(str(img_path))
        
        if image is None:
            print(f"Warning: Failed to load {img_path}")
            results_list.append(create_detection_output(image_id, []))
            continue
        
        # Detect with TTA
        bboxes = detect_with_tta(
            model, image, 
            conf=args.conf, 
            iou=args.iou, 
            imgsz=args.imgsz
        )
        
        # Convert to list of lists
        bboxes = [[float(x) for x in box] for box in bboxes]
        
        output_entry = create_detection_output(image_id, bboxes)
        results_list.append(output_entry)
    
    save_json(results_list, args.output)
    
    total_qr = sum(len(r['qrs']) for r in results_list)
    avg_qr = total_qr / len(results_list) if results_list else 0
    
    print(f"\n✓ Results saved to: {args.output}")
    print(f"✓ Total images: {len(results_list)}")
    print(f"✓ Total QR codes: {total_qr}")
    print(f"✓ Average QR per image: {avg_qr:.2f}")
    
    return results_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced QR Detection")
    
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='outputs/submission_detection_1.json')
    parser.add_argument('--weights', type=str, default='weights/best.pt')
    parser.add_argument('--conf', type=float, default=0.15,
                        help='Lower confidence = detect more QR codes')
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=str, default='0')
    
    args = parser.parse_args()
    infer_enhanced(args)