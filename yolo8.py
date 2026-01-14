
import yaml
import os
import cv2

from ultralytics import YOLO
model = YOLO('yolov8/yolov8n.pt')
# print(model.names)

def train_from_datasets():
    # ======================== Training Code ==========================
    datadir = "C:/Users/Dark_Knight/Phd Projects/YOLOv8 Detection/datasets/"
    workingDir = "C:/Users/Dark_Knight/Phd Projects/YOLOv8 Detection/"

    numClasses = 5
    classes = ['Bus', 'Car', 'Motorcycle', 'Pickup', 'Truck']

    file_dict = {
        'train' : os.path.join(datadir, "train"),
        'val' : os.path.join(datadir, "valid"),
        'test' : os.path.join(datadir, "test"),
        'nc': numClasses,
        'names': classes
    }

    yamlFile = os.path.join(workingDir, 'data.yaml')
    with open(yamlFile, 'w+') as f:
        yaml.dump(file_dict, f)

    model.train(data = yamlFile, epochs=30, imgsz=640)


def detect_from_image(imgFileName, isSave = False):
    # data1 = model.predict("image2.png", save=True)
    data = model.predict(imgFileName, classes=[2,5,7])

    points = []
    boxes = data[0].boxes.xyxy
    for box in boxes:
        data = box.tolist()
        start_point = (int(data[0]), int(data[1]))
        end_point = (int(data[2]), int(data[3]))
        points.append([start_point[0], start_point[1], end_point[0], end_point[1]])

    if isSave:
        img = cv2.imread(imgFileName)
        for point in points:
            img = cv2.rectangle(img, point[0:2], point[2:], (255,0,0), 2)
        cv2.imwrite(imgFileName.replace(imgFileName.split(".")[0], imgFileName.split(".")[0] + "_new"), img)

    return points


if __name__ == "__main__":
    detect_from_image("image4.png", True)