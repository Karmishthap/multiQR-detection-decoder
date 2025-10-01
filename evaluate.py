"""Evaluation script for QR detection"""
import argparse
import numpy as np
from pathlib import Path
from src.utils import load_json, calculate_iou

def calculate_metrics(predictions, ground_truth, iou_threshold=0.5):
    """Calculate precision, recall, F1 for QR detection"""
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Match predictions to ground truth
    for pred_entry in predictions:
        img_id = pred_entry['image_id']
        pred_boxes = [qr['bbox'] for qr in pred_entry['qrs']]
        
        # Find corresponding ground truth
        gt_entry = next((g for g in ground_truth if g['image_id'] == img_id), None)
        if gt_entry is None:
            total_fp += len(pred_boxes)
            continue
        
        gt_boxes = [qr['bbox'] for qr in gt_entry['qrs']]
        
        # Match pred to GT
        matched_gt = set()
        tp = 0
        
        for pred_box in pred_boxes:
            best_iou = 0
            best_idx = -1
            
            for idx, gt_box in enumerate(gt_boxes):
                if idx in matched_gt:
                    continue
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_idx)
            else:
                total_fp += 1
        
        total_tp += tp
        total_fn += len(gt_boxes) - len(matched_gt)
    
    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }

def evaluate(args):
    """Evaluate predictions against ground truth"""
    
    print(f"Loading predictions from: {args.predictions}")
    predictions = load_json(args.predictions)
    
    print(f"Loading ground truth from: {args.ground_truth}")
    ground_truth = load_json(args.ground_truth)
    
    print(f"\nEvaluating with IoU threshold: {args.iou_threshold}")
    metrics = calculate_metrics(predictions, ground_truth, args.iou_threshold)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"\nTrue Positives:  {metrics['tp']}")
    print(f"False Positives: {metrics['fp']}")
    print(f"False Negatives: {metrics['fn']}")
    print("="*50)
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QR Detection")
    
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions JSON')
    parser.add_argument('--ground_truth', type=str, required=True,
                        help='Path to ground truth JSON')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='IoU threshold for matching')
    
    args = parser.parse_args()
    evaluate(args)