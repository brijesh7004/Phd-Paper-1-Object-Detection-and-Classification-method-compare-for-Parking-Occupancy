from tkinter import *
from tkinter import ttk
import tkinter.filedialog as fd
from PIL import Image, ImageTk
import cv2 
import numpy as np

listColumns = ("Sr.", "P1 (TL)", "P2 (TR)", "P3 (BR)", "P4 (BL)")
mainShape = None;   resizeShape = None
points = [];    numPoints = 1;  totalConfigPoints = 1
mainImg = [];   resizeImg = []

def select_video_file():
    filetypes = (('Video files', '*.mp4'), ('All files', '*.*'))
    filename = fd.askopenfilename(title='Open a file', initialdir='/', filetypes=filetypes)   
    if filename is None or filename == "":
        return
    strVideoFileName.set(filename)
    # print(strVideoFileName.get())
    getFirstFrameFromVideo()
    
def getFirstFrameFromVideo():
    vidObj = cv2.VideoCapture(strVideoFileName.get())
    success, image = vidObj.read()
    while not success:
        success, image = vidObj.read()

    global mainImg, resizeImg
    mainImg = image        
    findNewImageShape(image.shape, 750) # print(resizeShape)
    resizeImg = cv2.resize(image, resizeShape, fx = 0.1, fy = 0.1)
    cv2.imwrite("sample.jpg", resizeImg)     
    loadImageOnCanvas()

def loadImageOnCanvas():
    img = ImageTk.PhotoImage(file="sample.jpg")
    photo.create_image(0, 0, anchor="nw", image=img)
    # photo.config(image=img)
    photo.image = img

def select_config_file():
    if strVideoFileName.get() != "None":
        filetypes = (('config files', '*.config'), ('text files', '*.txt'), ('All files', '*.*'))    
        filename = fd.askopenfilename(title='Open a Config file', initialdir='/', filetypes=filetypes)   
        if filename is None or filename == "":
            return
        strConfigFileName.set(filename)
        load_config_file()    

def load_config_file():
    if strConfigFileName.get() != "None":
        loadImageOnCanvas()

        global totalConfigPoints
        totalConfigPoints = 1

        for item in treeview.get_children():
            treeview.delete(item)
        
        f = open(strConfigFileName.get(), "r") 
        for line in f.readlines():
            v = line.replace('\n','').split(",")
            if len(v)==9:
                points = [totalConfigPoints, 
                            "{},{}".format(v[1], v[2]),
                            "{},{}".format(v[3], v[4]),
                            "{},{}".format(v[5], v[6]),
                            "{},{}".format(v[7], v[8])]
                treeview.insert('', END, values=points)
                totalConfigPoints = totalConfigPoints + 1
                drawAreaOnImage(v[1:])      
        f.close()
    else:
        print("Config not loaded")
    # treeview.insert('', END, values=[2, "12", "!2", "@3", "#$"])

def start_video_file(): 
    global mainImg
    counter = 1
    for item in treeview.get_children():
        v = treeview.item(item)['values']
        v2 = []
        for item in v:
            if str(item).__contains__(","):
                tmp = item.split(",")
                v2.append([int(tmp[0]), int(tmp[1])])
        cropImageFromCanvas(mainImg, v2, counter)
        counter = counter+1
        # file.write(("{},{},{},{},{}\n").format(v[0], v[1], v[2], v[3], v[4]))

def get_image_xy(event):
    global numPoints, points, totalConfigPoints
    x = event.x;    y = event.y
    sp = getImageShapeS1toS2(x,y, resizeShape, mainShape)
    # print(f"Clicked at coordinates (x, y): ({x}, {y}), Main Dimension: {sp}")
    if numPoints > 0:
        points.append(format("{},{}").format(sp[0], sp[1]))
        numPoints = numPoints-1
    if (len(points)) == 5:
        treeview.insert('', END, values=points)
        totalConfigPoints = totalConfigPoints + 1
    
        v = []
        for item in points:
            if str(item).__contains__(","):
                v  = v + item.split(",")
        drawAreaOnImage(v)
        points = []

def init_image_xy():   
    if strVideoFileName.get() != "None":
        global numPoints, points
        numPoints = 4
        points = [totalConfigPoints]
    else:
        print("Video frame not available")

def save_config_file():
    if len(treeview.get_children())>0:
        filename = fd.asksaveasfile(mode='w', defaultextension=".config")
        if filename is None or filename.name == "": # asksaveasfile return `None` if dialog closed with "cancel".
            return
            
        file = open(filename.name, "w")    
        for item in treeview.get_children():
            v = treeview.item(item)['values']
            file.write(("{},{},{},{},{}\n").format(v[0], v[1], v[2], v[3], v[4]))
        file.close()

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

def drawAreaOnImage(v):
    global mainShape, resizeShape
    v2 = [int(i) for i in v]
    for i in range(0,8,2):
        tmp = getImageShapeS1toS2(v2[i], v2[i+1], mainShape, resizeShape)
        v2[i], v2[i+1] = tmp[0], tmp[1]
    photo.create_polygon(v2, fill="", outline="red", width=2)

def cropImageFromCanvas(img, point, id):
    points = np.array([point])
    rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
    cropped = img[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]].copy()
    cv2.imwrite("{}.jpg".format(id), cropped)

def window_close_handler():
    print("Window is being closed")
    save_main_configuration()
    root.destroy()

def load_main_configuration():
    try:
        file = open("config.dat", "r")
        strs = file.readlines()
        strVideoFileName.set(strs[0].replace("\n",""))
        if strVideoFileName.get().endswith(".mp4"):
            getFirstFrameFromVideo()
        strConfigFileName.set(strs[1].replace("\n",""))
        if strConfigFileName.get().endswith(".config"):
            load_config_file()
        file.close()
    except:
        print("File not found or empty")

def save_main_configuration():
    file = open("config.dat", "w")
    file.write(strVideoFileName.get() + "\n")
    file.write(strConfigFileName.get())
    file.close()


 
root = Tk()
root.title('Parking Occupasion')
root.resizable(False, False)
root.geometry("1100x600")
root.protocol("WM_DELETE_WINDOW", window_close_handler)

strVideoFileName = StringVar()
strVideoFileName.set("None")
strConfigFileName = StringVar()
strConfigFileName.set("None")
p = PhotoImage()

#================== Create Top File Select Frame ======================
top_frame = Frame(root, width=1000, height=50, bg='lightgrey')
top_frame.pack(fill=X, side=TOP) #grid(row=0, column=0, columnspan=10, padx=5, pady=5)

Label(top_frame, text="Video : ", width=10).grid(row=0, column=0, padx=5, pady=2)
lblFileName = Label(top_frame, textvariable =strVideoFileName, width=120, justify="left").grid(row=0, column=1, columnspan=7, padx=2, pady=2)
b1 = Button(top_frame, text='Load', width=10, command=select_video_file).grid(row=0, column=8, padx=2, pady=2)
b2 = Button(top_frame, text='Start', width=10, command=start_video_file).grid(row=0, column=9, padx=2, pady=2)

Label(top_frame, text="Config : ", width=10).grid(row=1, column=0, padx=5, pady=2)
lblFileName = Label(top_frame, textvariable =strConfigFileName, width=120).grid(row=1, column=1, columnspan=7, padx=2, pady=2)
b3 = Button(top_frame, text='Load', width=10, command=select_config_file).grid(row=1, column=8, padx=2, pady=2)
b4 = Button(top_frame, text='Reload', width=10, command=load_config_file).grid(row=1, column=9, padx=2, pady=2)


#===================== Create Bottom File Select Frame =============================
bottom_frame = Frame(root, width=1000, height=550, bg='lightgrey')
bottom_frame.pack(fill=BOTH, side=LEFT, expand=True, padx=5, pady=5)

Button(bottom_frame, text='Add X-Y Co-ordinate', width=25, command=init_image_xy).place(x=0, y=0)
Button(bottom_frame, text='Save Config', width=20, command=save_config_file).place(x=195, y=0)

# Co-ordinate List Frame
treeFrame = Frame(bottom_frame)
treeFrame.place(x=0, y=30)
treeScroll = Scrollbar(treeFrame)
treeScroll.pack(side="right", fill="y")

treeview = ttk.Treeview(treeFrame, show="headings",
                        yscrollcommand=treeScroll.set, columns=listColumns, height=25)
for col_name in listColumns:
    treeview.heading(col_name, text=col_name)
    treeview.column(col_name, width=(10 if col_name == "Sr." else 80))
treeview.pack()
treeScroll.config(command=treeview.yview)

# Image Frame
# imageFrame = Frame(bottom_frame)
# imageFrame.place(x=350, y=0)
# imageScroll1 = Scrollbar(imageFrame, 'vertical')
# imageScroll1.pack(side="right", fill="y")
# imageScroll2 = Scrollbar(imageFrame, 'horizontal')
# imageScroll2.pack(side="bottom", fill="x")
photo = Canvas(bottom_frame, borderwidth=1, bg="white", width=750, height=500)
photo.place(x=350, y=0)
photo.bind("<Control-Button-1>", get_image_xy)

load_main_configuration()
root.mainloop()