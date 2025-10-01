"""JSON input/output utilities"""
import json
from pathlib import Path

def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json(data, filepath):
    """Save data to JSON file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def create_detection_output(image_id, bboxes):
    """
    Create detection output in required format
    Args:
        image_id: str, image identifier
        bboxes: list of [x_min, y_min, x_max, y_max]
    Returns:
        dict: {"image_id": "...", "qrs": [{"bbox": [...]}, ...]}
    """
    return {
        "image_id": image_id,
        "qrs": [{"bbox": bbox} for bbox in bboxes]
    }

def parse_yolo_annotation(txt_path, img_width, img_height):
    """
    Parse YOLO format annotation file
    Format: class_id x_center y_center width height (normalized)
    Returns: list of [x_min, y_min, x_max, y_max] in pixel coordinates
    """
    bboxes = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                _, x_center, y_center, w, h = map(float, parts[:5])
                
                # Convert to pixel coordinates
                x_center *= img_width
                y_center *= img_height
                w *= img_width
                h *= img_height
                
                # Convert to xyxy
                x_min = int(x_center - w / 2)
                y_min = int(y_center - h / 2)
                x_max = int(x_center + w / 2)
                y_max = int(y_center + h / 2)
                
                bboxes.append([x_min, y_min, x_max, y_max])
    return bboxes

def parse_coco_annotation(json_path):
    """
    Parse COCO format annotation
    Returns: dict mapping image_id to list of bboxes
    """
    data = load_json(json_path)
    
    # Create image id to filename mapping
    img_map = {img['id']: img['file_name'] for img in data['images']}
    
    # Group annotations by image
    annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        img_name = img_map.get(img_id, str(img_id))
        
        # COCO format: [x, y, width, height]
        x, y, w, h = ann['bbox']
        bbox = [int(x), int(y), int(x + w), int(y + h)]
        
        if img_name not in annotations:
            annotations[img_name] = []
        annotations[img_name].append(bbox)
    
    return annotations