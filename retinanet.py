# Import necessary libraries
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
# from keras_retinanet.utils.visualization import draw_box, draw_caption
# from keras_retinanet.utils.colors import label_color
import numpy as np
import cv2
import time

# Load the pre-trained RetinaNet model
model = models.load_model('keras_retinanet/resnet50_coco_best_v2.1.0.h5')
labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}


def detect_from_image(imgFileName, isSave = False):
    # Read and preprocess the input image
    image = read_image_bgr(imgFileName)
    image = preprocess_image(image)
    image, scale = resize_image(image)
    image2 = cv2.imread(imgFileName)

    # Make predictions using the RetinaNet model
    time1 = time.time()
    predictions = model.predict_on_batch(np.expand_dims(image, axis=0))
    print(time.time()-time1)

    # Extract the bounding boxes and class labels from the predictions
    boxes = predictions[0][0]
    scores = predictions[1][0]
    labels = predictions[2][0]

    boundaries = []
    # Draw the bounding boxes and class labels on the image
    for box, score, label in zip(boxes, scores, labels):
        if score < 0.5 or label != 2:
            continue
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = map(lambda x: int(x / scale), (x1, y1, x2, y2))
        boundaries.append([x1, y1, x2, y2])

        if isSave:
            cv2.rectangle(image2, (x1,y1), (x2,y2), (0,0,255), 2)     
            img_path = str(imgFileName).split(".")[:-1] + "_frame.jpg"
            cv2.imwrite(img_path, image2)
            # draw_box(image, (x1, y1, x2, y2), color=label_color(label))
            # draw_caption(image, (x1, y1), '{:.2f} {}'.format(score, labels_to_names[label]))
    
    return boundaries
