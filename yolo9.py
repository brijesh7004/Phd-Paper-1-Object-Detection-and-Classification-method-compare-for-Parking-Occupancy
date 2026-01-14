# C:/Users/Dark_Knight/miniconda3/envs/venv_phd/python.exe "c:/Users/Dark_Knight/Phd Projects/YOLOv9 Detection/yolov9/detect.py" --weights 'gelan-e.pt' --source 'image1.jpg' --device 0
# C:/Users/Dark_Knight/miniconda3/envs/venv_phd/python.exe "c:/Users/Dark_Knight/Phd Projects/YOLOv9 Detection/yolov9/detect_dual.py" --weights 'yolov9-c.pt' --source 'image1.jpg' --device cpu
# --weights 'yolov9-c.pt' --source 'image1.jpg' --device cpu

import argparse
import yolov9
import yolov9.detect_dual
import yolov9.detect
import cv2
import os

def detect_from_image(imgFileName, isSave = False):
    resultFile = 'result/labels/' + imgFileName.split(".")[0] + '.txt'
    print(resultFile)
    try:
        os.remove(resultFile)
    except:
        print("File not found")

    weight='yolov9/gelan-c.pt'; device='cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=weight)
    parser.add_argument('--source', default=imgFileName, help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--device', default=device, help='maximum detections per image')
    parser.add_argument('--classes', default=[2, 5, 7], help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--nosave', default=True, help='do not save images/videos')
    parser.add_argument('--save-txt', default=True, help='save results to *.txt')
    # parser.add_argument('--save-crop', default=True, help='save cropped prediction boxes')
    parser.add_argument('--project', default='', help='save results to project/name')
    parser.add_argument('--name', default='result', help='save results to project/name')
    parser.add_argument('--exist-ok', default='True', help='existing project/name ok, do not increment')

    if weight.startswith('yolov9/yolo'):
        yolov9.detect_dual.main(parser.parse_args())
    elif weight.startswith('yolov9/gelan'):
        yolov9.detect.main(parser.parse_args())

    lines = []
    with open(resultFile, 'r') as f:        
        lines = f.readlines()

    points = []
    for line in lines:
        data = line.replace("\n","").split(" ")
        data = [int(i) for i in data]
        data = data[1:]
        points.append(data)

    if isSave:
        img = cv2.imread(imgFileName)
        for point in points:
            # print(point)
            img = cv2.rectangle(img, point[0:2], point[2:], (255,0,0), 2)
        cv2.imwrite(imgFileName.replace(imgFileName.split(".")[0], imgFileName.split(".")[0] + "_new"), img)

    return points


if __name__ == "__main__":
    detect_from_image("image2.png", True)