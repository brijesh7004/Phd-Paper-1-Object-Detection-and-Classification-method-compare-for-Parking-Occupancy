import numpy as np
import matplotlib.pyplot as plt
import time

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# from skimage import io
from keras.preprocessing import image

type = 'det' # det, cls
folder_det = ['Retinanet', 'SSD', 'Yolo8', 'Yolo9']
folder_cls = ['AlexNet', 'MobileNet', 'ResNet50', 'VGG16']

plt.ion()
fig = plt.figure()
for counter in range(200):
    print(counter)
    imgNum=1
    for folder in (folder_det if type=='det' else folder_cls):
        path = 'result_{}_image/{}/img_{}_{}_{}.jpg'.format(type, folder, type, counter, folder.lower())
        show_img=image.load_img(path, target_size=(300, 500))

        # ttl = '{} - {}'.format(folder, counter)
        # plt.title('Frame- {}'.format(counter))
        ax = fig.add_subplot(2,2,imgNum)        
        ax.set_title(folder)
        ax.imshow(show_img)
        ax.axis('off')
        imgNum += 1

    fig.canvas.draw()
    time.sleep(1)
    fig.canvas.flush_events()

