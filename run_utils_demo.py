import os
from src import utils
import numpy as np
import cv2

os.makedirs('outputs', exist_ok=True)

print('Testing bbox_utils...')
b = [10, 20, 110, 220]
xywh = utils.xyxy_to_xywh(b)
print('xyxy_to_xywh:', xywh)
rev = utils.xywh_to_xyxy(xywh)
print('xywh_to_xyxy:', rev)

box1 = [10, 20, 110, 220]
box2 = [50, 100, 150, 260]
iou = utils.calculate_iou(box1, box2)
print('IoU between box1 and box2:', iou)

boxes = [box1, box2, [12,22,108,218]]
scores = [0.9, 0.75, 0.85]
keep = utils.nms(boxes, scores, iou_threshold=0.3)
print('NMS keep indices:', keep)

print('\nTesting json_utils...')
out = utils.create_detection_output('image_001', [box1, box2])
print('create_detection_output:', out)

utils.save_json(out, 'outputs/demo_output.json')
loaded = utils.load_json('outputs/demo_output.json')
print('Loaded JSON from outputs/demo_output.json:', loaded)

print('\nTesting image_utils...')
# Create a dummy image
img = np.zeros((300, 400, 3), dtype=np.uint8)
img[:] = (50, 50, 80)
# Draw boxes
img_with_boxes = utils.draw_bboxes(img, [box1, box2], color=(0,255,0), thickness=2)
cv2.imwrite('outputs/img_with_boxes.jpg', img_with_boxes)
print('Wrote outputs/img_with_boxes.jpg')

cropped = utils.crop_bbox(img, box1, padding=5)
cv2.imwrite('outputs/cropped.jpg', cropped)
print('Wrote outputs/cropped.jpg')

resized, scale = utils.resize_with_aspect(img, target_size=200)
cv2.imwrite('outputs/resized.jpg', resized)
print('Wrote outputs/resized.jpg, scale:', scale)

print('\nAll tests completed.')
