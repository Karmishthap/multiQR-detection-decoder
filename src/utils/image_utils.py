"""Image processing utilities"""
import cv2
import numpy as np
from pathlib import Path

def load_image(image_path):
    """Load image from path"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return img

def get_image_files(directory, extensions=['.jpg', '.jpeg', '.png', '.bmp']):
    """Get all image files from directory"""
    directory = Path(directory)
    image_files = set()
    for ext in extensions:
        image_files.update(directory.glob(f'*{ext}'))
        image_files.update(directory.glob(f'*{ext.upper()}'))
    return sorted(list(image_files))

def draw_bboxes(image, bboxes, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on image"""
    img_copy = image.copy()
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), color, thickness)
    return img_copy

def crop_bbox(image, bbox, padding=0):
    """Crop image region defined by bbox with optional padding"""
    h, w = image.shape[:2]
    x_min, y_min, x_max, y_max = map(int, bbox)
    
    # Add padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    
    return image[y_min:y_max, x_min:x_max]

def resize_with_aspect(image, target_size=640):
    """Resize image maintaining aspect ratio"""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h)), scale