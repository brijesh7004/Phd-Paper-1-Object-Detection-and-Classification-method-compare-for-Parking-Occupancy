import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tkinter as tk, threading
from tkinter import ttk
import tkinter.filedialog as fd
from PIL import Image, ImageTk
import cv2
import numpy as np
import time

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import yolo9 as model
detection_Algo = "Yolo9"

detectionData = [['name', 'time', '1', '2', '3', '4']]
curImage = None

listColumns = ("Sr.", "Status")
vidObj = None
mainImg = [];   points = [];    points2 = [];   overlap = []
totalFrame = 0; imgIdx = -1; skipFrame = 500; vidFrames=0
curTime = time.time();  

strVideoFileName = ''; strConfigFileName = ''



def loadVideoFromFile():
    global vidObj, vidFrames, strVideoFileName
    vidObj = cv2.VideoCapture(strVideoFileName)   
    vidFrames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    print(vidFrames)

def getNextFrameFromVideo():
    global mainImg, vidObj, imgIdx
    success, image = vidObj.read()
    imgIdx = imgIdx + (1 if success else 0)
    print(imgIdx)
    while (imgIdx % skipFrame) != 0:
        success, image = vidObj.read()
        imgIdx = imgIdx + (1 if success else 0)
    
    mainImg = image        
    cv2.imwrite("main.jpg", mainImg)    
    return success   

# def loadImageOnCanvas():
#     global imgPlot, mainImg
#     # img = mpimg.imread('your_image.png')
#     imgPlot = plt.imshow(mainImg)
#     plt.show()

def start_video_file(): 
    global mainImg, points, points2, overlap, curTime, totalFrame, imgIdx, skipFrame, vidFrames, detectionData
    isSuccess = True
    
    while isSuccess:
        print('[Info] : {} / {}'.format(imgIdx, vidFrames))       
        boxes = model.detect_from_image('main.jpg', False)

        counter = 1
        for box in boxes:
            print(box)
            cropped = mainImg[box[1]: box[3], box[0]: box[2]].copy()
            cv2.imwrite("./tmpImg/img_det_{}_{}.jpg".format(int(imgIdx/skipFrame),counter), cropped)
            counter = counter + 1

        isSuccess = getNextFrameFromVideo()    

def load_main_configuration():
    global strVideoFileName, strConfigFileName
    try:
        file = open("config.dat", "r")
        strs = file.readlines()
        strVideoFileName = strs[0].replace("\n","")        
        print(strVideoFileName)
        if strVideoFileName.endswith(".mp4"):
            loadVideoFromFile()
            getNextFrameFromVideo()
        file.close()
    except:
        print("File not found or empty")
        

load_main_configuration()
start_video_file()