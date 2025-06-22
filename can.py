import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util, visualization_utils as viz_utils

# Load the saved model
print("Loading model...")
detect_fn = tf.saved_model.load('exported_model/saved_model')  # path to your exported model

# Load label map
label_map_path = 'label_map.pbtxt'  # Define gesture ID-to-label map
category_index = label_map_util.create_category_index_from_labelmap(label_map_path)

# Canvas settings
canvas = None
drawing = False
last_x, last_y = None, None

def detect_hand(frame):
    input_tensor = tf.convert_to_tensor([frame])
    detections = detect_fn(input_tensor)

    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    for i in range(len(scores)):
        if scores[i] > 0.7 and category_index[classes[i]]['name'] == 'index_finger':  # your gesture label
            box = boxes[i]
            h, w, _ = frame.shape
            y1, x1, y2, x2 = box
            x_center = int((x1 + x2) / 2 * w)
            y_center = int((y1 + y2) / 2 * h)
            return (x_center, y_center)
    return None

# Start webcam
cap = cv2.VideoCapture(0)

print("Starting Air Canvas...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    hand_pos = detect_hand(frame)

    if hand_pos:
        x, y = hand_pos
        if last_x is not None:
            cv2.line(canvas, (last_x, last_y), (x, y), (255, 0, 0), 5)
        last_x, last_y = x, y
        cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
    else:
        last_x, last_y = None, None

    combined = cv2.add(frame, canvas)
    cv2.imshow("Air Canvas - Faster R-CNN", combined)

    key = cv2.waitKey(1)
    if key == ord('c'):  # Clear canvas
        canvas = np.zeros_like(frame)
    elif key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
