import numpy as np
import argparse
import time
import cv2
import yaml
import logging
import os

# config file path
root = "ssd/"
CONFIG_FILE= root + "config.yml"

# JSON extract of configs
config = None
net = None

# VOC0712 classes. SSD model is trained on VOC
# CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
#            "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", 
#            "motorbike", "person", "pottedplant", "sheep", "sofa", "train", 
#            "tvmonitor"]
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", 
           "motorbike", "person", "pottedplant", "sheep", "sofa", "train", 
           "tvmonitor"]
DetectionClass = [7]


# Colour of the bounding box
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
logging.info("loading model…")

prototxt = root + "MobileNetSSD_deploy.prototxt.txt"
model = root + "MobileNetSSD_deploy.caffemodel"
globalConfidence = 0.3


"""[summary]
    Configurations are in yaml file. Load the config file.
    Returns: [JSON] -- []
"""
def open_yaml():
    global CONFIG_FILE 
    with open(CONFIG_FILE, "r") as data:
        try:
            config = yaml.load(data, Loader=yaml.FullLoader)
            return(config)
        except yaml.YAMLError as exc:
            print(exc)
            os.exit(1)
        except Exception as e:
            print(e)
            os.exit(1)


def load_model_config():
    global config, net
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    config = open_yaml()
    net = cv2.dnn.readNetFromCaffe(prototxt, model)


def ssd_detector(image):
    image = cv2.imread(image)
    (h, w) = image.shape[:2]
     
    '''
    load the input image and construct an input blob (it is collection of 
    single binary data stored in some database system) for the image and then 
    resize it to a fixed 300*300 pixels and after that, normalize the images 
    (note: normalization is done via the authors of MobileNet SSD implementation)
    '''
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, 
                                  (300, 300), 127.5)
     
    logging.info("computing object detections…")
    start = time.time()
     
    '''
    pass the blob through the network and compute the forward pass to detect 
    the objects and predictions
    '''
    net.setInput(blob)
    detections = net.forward()
     
    end = time.time() - start
    logging.info("SSD took: {:.6f}".format(end))
         
    '''
    loop over all the detection and extract the confidence score for each 
    detection. Filter out all the weak detections whose probability is less 
    than 20%. Print the detected object and their confidence score 
    (it tells us that how confident the model is that box contains an object 
    and also how accurate it is). It is calculated as 
     
    confident scores= probability of an object * IOU 
    IOU stands for Intersection over union.IOU= area of overlap / area of union
    '''
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence >= globalConfidence:
            idx = int(detections[0, 0, i, 1])
            if idx in DetectionClass:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)     
                print("[INFO] {}".format(label))
                cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)     
                y = startY - 15 if startY - 15 > 15 else startY + 15     
                cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            
    result_dir = config["result_dir"]
    img_path = result_dir + "frame.jpg"
    cv2.imwrite(img_path, image)
#cv2.waitKey(0)


def detect_from_image(image, isSave = False):
    boundaries = []
    image = cv2.imread(image)
    (h, w) = image.shape[:2]
     
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
     
    logging.info("computing object detections…")
    start = time.time()

    net.setInput(blob)
    detections = net.forward()
     
    end = time.time() - start
    logging.info("SSD took: {:.6f}".format(end))
         
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence >= confidence:
            idx = int(detections[0, 0, i, 1])
            if idx in DetectionClass:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                boundaries.append([startX, startY, endX, endY])

                if isSave:
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)     
                    print("[INFO] {}".format(label))
                    cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)     
                    y = startY - 15 if startY - 15 > 15 else startY + 15     
                    cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                    result_dir = config["result_dir"]
                    img_path = result_dir + "frame.jpg"
                    cv2.imwrite(img_path, image)

    return boundaries


def is_ssd_detect(image, object):
    checkClass = CLASSES.index(object)

    image = cv2.imread(image)
    (h, w) = image.shape[:2]
     
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
     
    logging.info("computing object detections…")
    start = time.time()

    net.setInput(blob)
    detections = net.forward()
     
    end = time.time() - start
    logging.info("SSD took: {:.6f}".format(end))
         
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence >= confidence:
            idx = int(detections[0, 0, i, 1])
            if idx in [checkClass]:
                return True

    return False


if __name__ == "__main__":
    # arguments passed in command line
    load_model_config()
    # ssd_detector('image2.png')

    boundaries = detect_from_image('image2.png', True)
    print(boundaries)

    img = cv2.imread('image2.png')
    for box in boundaries:
        img = cv2.rectangle(img, box[0:2], box[2:], (255, 0, 0), 2)
    cv2.imshow("Image", img) 
    # cv2.waitKey(0)

    print(is_ssd_detect('sample.jpg', 'car'))

    cv2.destroyAllWindows()