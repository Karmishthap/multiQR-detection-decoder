#!/usr/bin/env python3
"""
Complete pipeline to generate QR code detection and decoding submission for all 50 test images.
This script reproduces the entire workflow from scratch.

Estimated processing times:
- Total pipeline: ~45 minutes
- Detection stage: 5-15 minutes
- Decoding stage: 30-45 minutes
- Installation: <1 minute

Times may vary based on hardware, internet connection, and system performance.
"""

import os
import sys
import subprocess
import argparse

def run_command(command, description):
    """Run a shell command and print status"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*50)

    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✓ Success!")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed with exit code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout[-500:])
        if e.stderr:
            print("STDERR:", e.stderr[-500:])
        return False

def main():
    parser = argparse.ArgumentParser(description="Complete QR Detection and Decoding Pipeline")
    parser.add_argument('--weights', type=str, default='yolov8n.pt',
                        help='Path to YOLOv8 weights (default: yolov8n.pt)')
    parser.add_argument('--input_dir', type=str, default='data/demo_images/QR_Dataset/test_images',
                        help='Input directory with test images')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for results')
    parser.add_argument('--padding', type=int, default=50,
                        help='Padding for QR cropping in decoding')

    args = parser.parse_args()

    print("🚀 Starting Complete QR Detection and Decoding Pipeline")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Weights: {args.weights}")
    print(f"Padding: {args.padding}")
    print("\n⏱️  Estimated total processing time: ~45 minutes")
    print("   - Detection stage: 5-15 minutes")
    print("   - Decoding stage: 30-45 minutes")
    print("   (Times may vary based on hardware and system performance)\n")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Install dependencies
    if not run_command("pip install ultralytics>=8.0.0", "Install ultralytics (quick)"):
        sys.exit(1)

    if not run_command("pip install -r requirements.txt", "Install other dependencies from requirements.txt (quick)"):
        sys.exit(1)

    # Step 2: Run detection inference
    detection_cmd = f"python infer_enhaced.py --input {args.input_dir} --output {args.output_dir}/submission_detection_1.json --weights {args.weights}"
    if not run_command(detection_cmd, "Run enhanced detection inference on all test images (5-15 minutes)"):
        sys.exit(1)

    # Step 3: Run enhanced decoding
    decoding_cmd = f"python decode_enhaced.py --input {args.output_dir}/submission_detection_1.json --image_dir {args.input_dir} --output {args.output_dir}/submission_decoding_2.json --padding {args.padding}"
    if not run_command(decoding_cmd, "Run enhanced decoding with multiple preprocessing methods (30-45 minutes)"):
        sys.exit(1)

    # Step 4: Validate results
    print(f"\n{'='*50}")
    print("📊 VALIDATION RESULTS")
    print('='*50)

    # Check detection output
    detection_file = f"{args.output_dir}/submission_detection_1.json"
    if os.path.exists(detection_file):
        print(f"✓ Detection output created: {detection_file}")
        # Could add more validation here
    else:
        print(f"✗ Detection output missing: {detection_file}")
        sys.exit(1)

    # Check decoding output
    decoding_file = f"{args.output_dir}/submission_decoding_2.json"
    if os.path.exists(decoding_file):
        print(f"✓ Decoding output created: {decoding_file}")
        # Could add more validation here
    else:
        print(f"✗ Decoding output missing: {decoding_file}")
        sys.exit(1)

    print(f"\n🎉 Pipeline completed successfully!")
    print(f"Results saved in: {args.output_dir}")
    print(f"- Detection: {detection_file}")
    print(f"- Decoding: {decoding_file}")

if __name__ == "__main__":
    main()
