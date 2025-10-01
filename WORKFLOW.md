# TODO: Generate Submission for All 50 Test Images

## Step 1: Install Dependencies
- [x] Install ultralytics: `pip install ultralytics`
- [x] Install other dependencies from requirements.txt: `pip install -r requirements.txt`

## Step 2: Run Detection Inference (Stage 1)
- [x] Run enhanced inference on all test images: `python infer_enhaced.py --input data/demo_images/QR_Dataset/test_images --output outputs/submission_detection_1.json --weights yolov8n.pt`
- [x] Verify that `outputs/submission_detection_1.json` contains detection results for all 50 images (img201 to img250)

## Step 3: Run Decoding Inference (Stage 2) - If Applicable
- [x] Run decoding script: `python decode_classify.py --input outputs/submission_detection_1.json --image_dir data/demo_images/QR_Dataset/test_images --output outputs/submission_decoding_2.json`
- [x] Verify that `outputs/submission_decoding_2.json` contains decoding results for all 50 images

## Step 4: Validate Output Format
- [x] Ensure Stage 1 output is a JSON array with 50 objects, each having "image_id" and "qrs" with bbox arrays
- [x] Ensure Stage 2 output is a JSON array with 50 objects, each having "image_id" and "qrs" with bbox and "value" fields

## Step 5: Improve Decoding Accuracy to 80%
- [x] Enhanced decoding script with multiple preprocessing methods (grayscale, binary, adaptive threshold, sharpen, denoise, contrast enhancement, median blur, bilateral filter, morphological operations, gamma correction, histogram equalization)
- [x] Added more rotation angles (45, 135, 225, 315 degrees) and scaling factors (1.5x, 2.0x, 2.5x, 3.0x)
- [x] Integrated QReader library for additional decoding attempts
- [x] Increased padding to 50 pixels for better cropping
- [x] Achieved 87.7% decoding accuracy (157/179 QR codes successfully decoded)
