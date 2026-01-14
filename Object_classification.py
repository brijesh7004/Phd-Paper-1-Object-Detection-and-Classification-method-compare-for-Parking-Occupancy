import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image

root = './Classification/'
model_name = ['model_resnet50.h5', 'model_vgg16.h5', 'model_alexnet.h5', 'model_mobilenet.h5']
vehicle_class = ['Car', 'Bike']
model = None

def is_object_classify(fileName, modelName, object):
    global model
    if model is None:
        model = load_model(root + 'model_' + modelName + '.h5')

    img = image.load_img(fileName, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x /= 255
    custom = model.predict(x)
    idx = np.argmax(custom[0])
    
    # print(vehicle_class[idx], custom[0][idx])
    return vehicle_class[idx]==object and custom[0][idx] > 0.6