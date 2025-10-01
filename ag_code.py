"""Super aggressive decoding - maximize success rate"""
import argparse
import cv2
import numpy as np
from pyzbar import pyzbar
from pathlib import Path
from src.utils import load_json, save_json, load_image, crop_bbox
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def enhance_qr_image(img):
    """Aggressive image enhancement for QR decoding"""
    enhanced_versions = []
    
    # Original
    enhanced_versions.append(img)
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Method 1: Otsu threshold
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    enhanced_versions.append(otsu)
    
    # Method 2: Inverted Otsu (for dark QR on light background)
    enhanced_versions.append(cv2.bitwise_not(otsu))
    
    # Method 3: Adaptive threshold - mean
    adaptive_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                          cv2.THRESH_BINARY, 15, 10)
    enhanced_versions.append(adaptive_mean)
    
    # Method 4: Adaptive threshold - gaussian
    adaptive_gauss = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 15, 10)
    enhanced_versions.append(adaptive_gauss)
    
    # Method 5: Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    enhanced_versions.append(morph)
    
    # Method 6: Sharpen
    kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel_sharp)
    enhanced_versions.append(sharpened)
    
    # Method 7: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    enhanced_versions.append(clahe_img)
    
    # Method 8: Bilateral filter + threshold
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    _, bilateral_thresh = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    enhanced_versions.append(bilateral_thresh)
    
    return enhanced_versions

def decode_super_aggressive(image_crop):
    """Try every possible method to decode QR"""
    
    # Get enhanced versions
    enhanced = enhance_qr_image(image_crop)
    
    # Try pyzbar on all versions
    for img in enhanced:
        try:
            decoded = pyzbar.decode(img)
            if decoded:
                return decoded[0].data.decode('utf-8', errors='ignore')
        except:
            pass
    
    # Try OpenCV QRCodeDetector on all versions
    detector = cv2.QRCodeDetector()
    for img in enhanced:
        try:
            if len(img.shape) == 2:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = img
            data, _, _ = detector.detectAndDecode(img_bgr)
            if data:
                return data
        except:
            pass
    
    # Try rotation on best threshold version
    _, best_thresh = cv2.threshold(
        cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY) if len(image_crop.shape) == 3 else image_crop,
        0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    for angle in [90, 180, 270]:
        h, w = best_thresh.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(best_thresh, M, (w, h), 
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        
        decoded = pyzbar.decode(rotated)
        if decoded:
            return decoded[0].data.decode('utf-8', errors='ignore')
        
        try:
            rotated_bgr = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
            data, _, _ = detector.detectAndDecode(rotated_bgr)
            if data:
                return data
        except:
            pass
    
    # Try upscaling if small
    h, w = image_crop.shape[:2]
    if h < 100 or w < 100:
        target_size = 200
        scale = target_size / min(h, w)
        upscaled = cv2.resize(image_crop, None, fx=scale, fy=scale, 
                             interpolation=cv2.INTER_CUBIC)
        
        decoded = pyzbar.decode(upscaled)
        if decoded:
            return decoded[0].data.decode('utf-8', errors='ignore')
        
        gray_up = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY) if len(upscaled.shape) == 3 else upscaled
        _, thresh_up = cv2.threshold(gray_up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        decoded = pyzbar.decode(thresh_up)
        if decoded:
            return decoded[0].data.decode('utf-8', errors='ignore')
    
    return ""

def process_image(entry, image_dir, padding):
    img_id = entry['image_id']
    
    # Find image with multiple extension attempts
    img_path = None
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP', '.tif', '.tiff']:
        test_path = Path(image_dir) / f"{img_id}{ext}"
        if test_path.exists():
            img_path = test_path
            break
    
    if img_path is None:
        return {"image_id": img_id, "qrs": [{"bbox": q['bbox'], "value": ""} for q in entry['qrs']]}, 0
    
    try:
        image = load_image(str(img_path))
    except:
        return {"image_id": img_id, "qrs": [{"bbox": q['bbox'], "value": ""} for q in entry['qrs']]}, 0
    
    qrs_decoded = []
    success = 0
    
    for qr in entry['qrs']:
        bbox = qr['bbox']
        
        # Crop with padding
        cropped = crop_bbox(image, bbox, padding=padding)
        
        # Super aggressive decode
        value = decode_super_aggressive(cropped)
        
        if value:
            success += 1
        
        qrs_decoded.append({'bbox': bbox, 'value': value})
    
    return {"image_id": img_id, "qrs": qrs_decoded}, success

def main(args):
    detections = load_json(args.input)
    
    results = []
    total_success = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_image, entry, args.image_dir, args.padding) 
                   for entry in detections]
        
        for future in tqdm(futures, desc="Super Aggressive Decode"):
            result, success = future.result()
            results.append(result)
            total_success += success
    
    results.sort(key=lambda x: x['image_id'])
    save_json(results, args.output)
    
    total_qr = sum(len(r['qrs']) for r in results)
    rate = (total_success / total_qr * 100) if total_qr > 0 else 0
    
    print(f"\n✓ Results: {args.output}")
    print(f"✓ QR codes: {total_qr}")
    print(f"✓ Decoded: {total_success} ({rate:.1f}%)")
    print(f"✓ Failed: {total_qr - total_success}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--output', default='outputs/submission_decoding_2.json')
    parser.add_argument('--padding', type=int, default=25, help='Increase if still failing')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    main(args)