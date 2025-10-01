"""Enhanced QR decoding with multiple preprocessing methods"""
import argparse
import cv2
import numpy as np
from pyzbar import pyzbar
from qreader import QReader
from pathlib import Path
from src.utils import load_json, save_json, load_image, crop_bbox
from tqdm import tqdm

def preprocess_for_qr(image):
    """Generate multiple preprocessed versions"""
    versions = [image]  # Original

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    versions.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    # Binary threshold (Otsu)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    versions.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    versions.append(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR))

    # Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    versions.append(sharpened)

    # Denoise
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    versions.append(denoised)

    # Contrast enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    versions.append(enhanced)

    # Median blur
    median = cv2.medianBlur(image, 3)
    versions.append(median)

    # Bilateral filter
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    versions.append(bilateral)

    # Morphological operations
    kernel_morph = np.ones((3,3), np.uint8)
    morph_open = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_morph)
    versions.append(cv2.cvtColor(morph_open, cv2.COLOR_GRAY2BGR))

    morph_close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_morph)
    versions.append(cv2.cvtColor(morph_close, cv2.COLOR_GRAY2BGR))

    # Gamma correction
    gamma = 1.5
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(image, table)
    versions.append(gamma_corrected)

    # Histogram equalization
    if len(image.shape) == 3:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        hist_eq = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        hist_eq = cv2.equalizeHist(image)
        hist_eq = cv2.cvtColor(hist_eq, cv2.COLOR_GRAY2BGR)
    versions.append(hist_eq)

    return versions

def decode_qr_aggressive(image_crop):
    """Try multiple decoding methods aggressively"""

    # Initialize QReader
    qreader = QReader()

    # Generate preprocessed versions
    versions = preprocess_for_qr(image_crop)

    # Try qreader on all versions
    for img in versions:
        decoded_text = qreader.detect_and_decode(image=img)
        if decoded_text:
            return decoded_text

    # Try pyzbar on all versions
    for img in versions:
        decoded_objects = pyzbar.decode(img)
        if decoded_objects:
            return decoded_objects[0].data.decode('utf-8', errors='ignore')

    # Try OpenCV detector on all versions
    qr_detector = cv2.QRCodeDetector()
    for img in versions:
        data, bbox, _ = qr_detector.detectAndDecode(img)
        if data:
            return data
    
    # Try with rotation corrections
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        if angle > 0:
            h, w = image_crop.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image_crop, M, (w, h))

            decoded_objects = pyzbar.decode(rotated)
            if decoded_objects:
                return decoded_objects[0].data.decode('utf-8', errors='ignore')

            data, bbox, _ = qr_detector.detectAndDecode(rotated)
            if data:
                return data

    # Try upscaling for small QR codes
    h, w = image_crop.shape[:2]
    scales = [1.0, 1.5, 2.0, 2.5, 3.0]
    for scale in scales:
        if scale > 1.0 or (h < 100 or w < 100):
            upscaled = cv2.resize(image_crop, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_CUBIC)

            decoded_objects = pyzbar.decode(upscaled)
            if decoded_objects:
                return decoded_objects[0].data.decode('utf-8', errors='ignore')

            data, bbox, _ = qr_detector.detectAndDecode(upscaled)
            if data:
                return data
    
    return ""

def decode_enhanced(args):
    """Enhanced decoding with aggressive preprocessing"""
    
    print(f"Loading detections from: {args.input}")
    detections = load_json(args.input)
    
    results_list = []
    success_count = 0
    total_qr = 0
    
    for entry in tqdm(detections, desc="Decoding QR codes"):
        img_id = entry['image_id']
        
        # Find image
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']:
            test_path = Path(args.image_dir) / f"{img_id}{ext}"
            if test_path.exists():
                img_path = test_path
                break
        
        if img_path is None:
            print(f"Warning: Image not found for {img_id}")
            qrs_decoded = [{"bbox": qr['bbox'], "value": ""} for qr in entry['qrs']]
            results_list.append({"image_id": img_id, "qrs": qrs_decoded})
            continue
        
        image = load_image(str(img_path))
        
        # Decode each QR
        qrs_decoded = []
        for qr in entry['qrs']:
            total_qr += 1
            bbox = qr['bbox']
            
            # Crop with larger padding
            cropped = crop_bbox(image, bbox, padding=args.padding)
            
            # Aggressive decode
            decoded_value = decode_qr_aggressive(cropped)
            
            if decoded_value:
                success_count += 1
            
            qrs_decoded.append({
                'bbox': bbox,
                'value': decoded_value
            })
        
        results_list.append({
            'image_id': img_id,
            'qrs': qrs_decoded
        })
    
    save_json(results_list, args.output)
    
    decode_rate = (success_count / total_qr * 100) if total_qr > 0 else 0
    print(f"\n✓ Results saved to: {args.output}")
    print(f"✓ Total QR codes: {total_qr}")
    print(f"✓ Successfully decoded: {success_count} ({decode_rate:.1f}%)")
    
    return results_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced QR Decoding")
    
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='outputs/submission_decoding_2.json')
    parser.add_argument('--padding', type=int, default=20,
                        help='Padding around QR crop (increase if decoding fails)')
    
    args = parser.parse_args()
    decode_enhanced(args)