"""Stage 2: QR code decoding"""
import argparse
import cv2
from pyzbar import pyzbar
from pathlib import Path
from src.utils import load_json, save_json, load_image, crop_bbox
from tqdm import tqdm

def decode_qr(image_crop):
    """Decode QR code from image crop using pyzbar and OpenCV"""
    
    # Try with pyzbar first
    decoded_objects = pyzbar.decode(image_crop)
    if decoded_objects:
        return decoded_objects[0].data.decode('utf-8', errors='ignore')
    
    # Try with OpenCV QRCodeDetector
    qr_detector = cv2.QRCodeDetector()
    data, bbox, _ = qr_detector.detectAndDecode(image_crop)
    
    if data:
        return data
    
    return ""

def decode_qr_codes(args):
    """Decode QR codes from detection results"""
    
    # Load detection results
    print(f"Loading detections from: {args.input}")
    detections = load_json(args.input)
    
    results_list = []
    success_count = 0
    total_qr = 0
    
    for entry in tqdm(detections, desc="Decoding QR codes"):
        img_id = entry['image_id']
        
        # Find image file
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG']:
            test_path = Path(args.image_dir) / f"{img_id}{ext}"
            if test_path.exists():
                img_path = test_path
                break
        
        if img_path is None:
            print(f"Warning: Image not found for {img_id}")
            # Add empty values for missing images
            qrs_decoded = [{"bbox": qr['bbox'], "value": ""} for qr in entry['qrs']]
            results_list.append({"image_id": img_id, "qrs": qrs_decoded})
            continue
        
        image = load_image(str(img_path))
        
        # Decode each QR code
        qrs_decoded = []
        for qr in entry['qrs']:
            total_qr += 1
            bbox = qr['bbox']
            
            # Crop QR region with padding
            cropped = crop_bbox(image, bbox, padding=10)
            
            # Decode
            decoded_value = decode_qr(cropped)
            
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
    
    # Save results
    save_json(results_list, args.output)
    
    print(f"\n✓ Results saved to: {args.output}")
    print(f"✓ Total QR codes: {total_qr}")
    print(f"✓ Successfully decoded: {success_count} ({success_count/total_qr*100:.1f}%)")
    
    return results_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Decode QR Codes")
    
    parser.add_argument('--input', type=str, required=True,
                        help='Path to detection JSON (submission_detection_1.json)')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Path to images directory')
    parser.add_argument('--output', type=str, default='outputs/submission_decoding_2.json',
                        help='Output JSON file path')
    
    args = parser.parse_args()
    decode_qr_codes(args)