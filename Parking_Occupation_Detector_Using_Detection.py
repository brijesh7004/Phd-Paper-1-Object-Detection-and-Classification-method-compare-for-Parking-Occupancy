import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tkinter as tk, threading
from tkinter import ttk
import tkinter.filedialog as fd
from PIL import Image, ImageTk
import cv2
import numpy as np
import time

model_name = ['yolo8', 'yolo9', 'ssd', 'retinanet']
print(model_name)

modelIdx = int(input("Enter Model No: "))
detection_Algo = model_name[modelIdx-1]
if detection_Algo == "yolo8":
    import yolo8 as model
elif detection_Algo == "yolo9":
    import yolo9 as model
elif detection_Algo == "ssd":
    import ssd as model
    model.load_model_config()
elif detection_Algo == "retinanet":
    import retinanet as model


detectionData = [['name', 'time', '1', '2', '3', '4']]
curImage = None


listColumns = ("Sr.", "Status")
mainShape = None;   resizeShape = None;     vidObj = None
points = [];    points2 = [];   overlap = []
mainImg = [];   resizeImg = []
totalFrame = 0; imgIdx = -1;    skipFrame = 500; vidFrames=0
curTime = time.time();  

def loadVideoFromFile():
    global vidObj, vidFrames
    vidObj = cv2.VideoCapture(strVideoFileName.get())   
    vidFrames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    print(vidFrames)

def getNextFrameFromVideo(isUpdate=False):
    global mainImg, resizeImg, vidObj, imgIdx
    success, image = vidObj.read()
    imgIdx = imgIdx + (1 if success else 0)
    while (imgIdx % skipFrame) != 0:
        success, image = vidObj.read()
        imgIdx = imgIdx + (1 if success else 0)
    
    mainImg = image        
    findNewImageShape(image.shape, 750) # print(resizeShape)
    resizeImg = cv2.resize(image, resizeShape, fx = 0.1, fy = 0.1)
    cv2.imwrite("sample.jpg", resizeImg)
    cv2.imwrite("main.jpg", mainImg)
    if isUpdate:
        loadImageOnCanvas()
    return success

def loadImageOnCanvas():
    img = ImageTk.PhotoImage(file="sample.jpg")
    photo.create_image(0, 0, anchor="nw", image=img)
    # photo.config(image=img)
    photo.image = img

def load_config_file():
    if strConfigFileName.get() != "None":
        loadImageOnCanvas()

        global points, points2

        for item in treeview.get_children():
            treeview.delete(item)
        
        f = open(strConfigFileName.get(), "r") 
        for line in f.readlines():
            v = line.replace('\n','').split(",")
            if len(v)==9:
                p = []
                for item in v:
                    p.append(int(item))
                points.append(p)
                points2.append([min(p[1::2]),min(p[2::2]),max(p[1::2]),max(p[2::2])])
                treeview.insert('', tk.END, values=[int(p[0]), "-----"])
        f.close()
        drawROIonImage()
    else:
        print("Config not loaded")

def drawROIonImage():
    global points
    # print(len(points))
    for item in points:
        drawAreaOnImage(item[1:], 'black')

def drawAreaOnImage(v, color):
    global mainShape, resizeShape, mainImg
    v2 = [int(i) for i in v]
    v3 = []
    for i in range(0,8,2):
        v3.append([v2[i], v2[i+1]])
        tmp = getImageShapeS1toS2(v2[i], v2[i+1], mainShape, resizeShape)
        v2[i], v2[i+1] = tmp[0], tmp[1]        
    photo.create_polygon(v2, fill="", outline=color, width=2)

    pts = np.array(v3, np.int32)
    mainImg = cv2.polylines(mainImg, [pts.reshape((-1, 1, 2))], True, (0,255,0) if color=='green' else (0,0,255), 2)

def drawAreaOnImage2(v, color, isUpdate):
    if not isUpdate:
        return
    global mainShape, resizeShape
    v[0], v[1] = getImageShapeS1toS2(v[0], v[1], mainShape, resizeShape)
    v[2], v[3] = getImageShapeS1toS2(v[2], v[3], mainShape, resizeShape)
    photo.create_rectangle(v[0], v[1], v[2], v[3], fill="", outline=color, width=1)
    v[0], v[1] = getImageShapeS1toS2(v[0], v[1], resizeShape, mainShape)
    v[2], v[3] = getImageShapeS1toS2(v[2], v[3], resizeShape, mainShape)


def start_video_file(): 
    global mainImg, points, points2, overlap, curTime, totalFrame, imgIdx, skipFrame, vidFrames, detectionData
    isSuccess = True
    
    while isSuccess:
        name = 'img_det_{}_{}.jpg'.format(int(imgIdx/skipFrame), detection_Algo)
        dt, name = [name], './result_det_image/' + name
        print('[Info] : {} / {}'.format(imgIdx, vidFrames))

        if totalFrame==0:
            curTime = time.time()

        time1 = time.time()
        boxes = model.detect_from_image('main.jpg', False)
        dt.append(time.time()-time1)

        for item in treeview.get_children():
            treeview.delete(item)

        # for p in points2:
        #     print(p)
        # for b in boxes:
        #     print(b)

        loadImageOnCanvas()
        counter, overlap, updateYolo = 0, [], True
        for point in points2:
            valid, p = 0, points[counter]
            for box in boxes:
                # drawAreaOnImage2(box, 'blue', updateYolo)
                tmp = getOverlapPerc(point, box, p[1:])
                if tmp > valid:
                    valid = tmp
            
            updateYolo = False
            treeview.insert('', tk.END, values=[counter+1, 'Occupied' if valid > 70 else 'Free'])
            drawAreaOnImage(p[1:], 'red' if valid > 70 else 'green')         
            counter = counter + 1
            dt.append(1 if valid > 70 else 0)

            totalFrame = totalFrame + 1
            if totalFrame == 10:
                tmp = time.time()
                root.title('Parking Occupasion ({0}), {1:3.2} FPS'.format(detection_Algo, totalFrame/(tmp-curTime)))
                totalFrame = 0

        detectionData.append(dt)
        cv2.imwrite(name, mainImg)

        file = open("./result_csv/data_det_{}.csv".format(detection_Algo), "w")
        for item in detectionData:
            item2 = [itm if type(itm)==str else str(itm) for itm in item]
            file.write(",".join(item2) + "\n")
        file.close()

        isSuccess = getNextFrameFromVideo()    

def cropImageFromCanvas(img, point, id):
    points = np.array([point])
    rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
    cropped = img[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]].copy()
    cv2.imwrite("{}.jpg".format(id), cropped)

def findNewImageShape(sz, newWidth):
    # print(sz)    
    global mainShape, resizeShape
    if sz[0] > sz[1]:
        mainShape = (sz[0], sz[1])
        resizeShape = (int(newWidth), int(sz[1]*newWidth/sz[0]))
    else:    
        mainShape = (sz[1], sz[0])
        resizeShape = (int(newWidth), int(sz[0]*newWidth/sz[1]))

def getImageShapeS1toS2(w, h, s1, s2):
    return (int(w*s2[0]/s1[0]), int(h*s2[1]/s1[1]))

def load_main_configuration():
    try:
        file = open("config.dat", "r")
        strs = file.readlines()
        strVideoFileName.set(strs[0].replace("\n",""))
        if strVideoFileName.get().endswith(".mp4"):
            loadVideoFromFile()
            getNextFrameFromVideo(True)
        strConfigFileName.set(strs[1].replace("\n",""))
        if strConfigFileName.get().endswith(".config"):
            load_config_file()
        file.close()
    except:
        print("File not found or empty")

def getOverlapPerc(p1,p2,p3):
    diffX, diffY = abs(p1[2]-p1[0])*0.2, abs(p1[3]-p1[1])*0.2

    if (((abs(p1[0]-p2[0]) < diffX) or p1[0]>p2[0]) 
        and ((abs(p1[1]-p2[1]) < diffY) or p1[1]>p2[1])
            and (abs(p1[2]-p2[2]) < diffX or p1[2]<p2[2]) 
                and (abs(p1[3]-p2[3]) < diffY or p1[3]<p2[3])):
        return 100
    elif (((abs(p3[0]-p2[0]) < diffX) or p3[0]>p2[0]) 
        and ((abs(p3[1]-p2[1]) < diffY) or p3[1]>p2[1])
            and (abs(p3[2]-p2[2]) < diffX or p3[2]<p2[2]) 
            and (abs(p3[3]-p2[1]) < diffY or p3[3]>p2[1])
                and (abs(p3[4]-p2[2]) < diffX or p3[4]<p2[2]) 
                and (abs(p3[5]-p2[3]) < diffY or p3[5]<p2[3])
                    and (abs(p3[6]-p2[0]) < diffX or p3[6]>p2[0]) 
                    and (abs(p3[7]-p2[3]) < diffY or p3[7]<p2[3])):
        return 100
    elif (abs(p1[0]-p2[0]) < diffX) and (abs(p1[1]-p2[1]) < diffY):
        p_overlap = [max(p1[0], p2[0]), max(p1[1], p2[1]), min(p1[2], p2[2]), min(p1[3], p2[3])]
        area_overlap = (p_overlap[3]-p_overlap[1]) * (p_overlap[2]-p_overlap[0])
        area_total = (p1[3]-p1[1]) * (p1[2]-p1[0])
        # print(area_total, area_overlap, area_overlap/area_total*100)
        return int(area_overlap/area_total*100)
    else:
        return 0

 
root = tk.Tk()
root.title('Parking Occupasion')
root.resizable(False, False)
root.geometry("900x520")

strVideoFileName = tk.StringVar()
strVideoFileName.set("None")
strConfigFileName = tk.StringVar()
strConfigFileName.set("None")
p = tk.PhotoImage()

#===================== Create Bottom File Select Frame =============================
bottom_frame = tk.Frame(root, width=1000, height=550, bg='lightgrey')
bottom_frame.pack(fill=tk.BOTH, side=tk.LEFT, expand=True, padx=5, pady=5)

# Co-ordinate List Frame
treeFrame = tk.Frame(bottom_frame)
treeFrame.place(x=0, y=0)
treeScroll = tk.Scrollbar(treeFrame)
treeScroll.pack(side="right", fill="y")

treeview = ttk.Treeview(treeFrame, show="headings",
                        yscrollcommand=treeScroll.set, columns=listColumns, height=24)
for col_name in listColumns:
    treeview.heading(col_name, text=col_name)
    treeview.column(col_name, width=(10 if col_name == "Sr." else 100))
treeview.pack()
treeScroll.config(command=treeview.yview)

photo = tk.Canvas(bottom_frame, borderwidth=1, bg="white", width=750, height=500)
photo.place(x=130, y=0)

load_main_configuration()

thread = threading.Thread(target=start_video_file, args=())
thread.daemon = 0.1
thread.start()

root.mainloop()
