import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.applications import ResNet50

# from skimage import io
from keras.preprocessing import image

model_name = ['model_resnet50.h5', 'model_vgg16.h5', 'model_alexnet.h5', 'model_mobilenet.h5']
loc = [[0,0], [0,1], [1,0], [1,1]]
path = 'check_car.png'
# fig, axs = plt.subplots(2, 2)
fig = plt.figure()
counter = 1

for mdl in model_name:
    model = load_model(mdl)

    img = image.load_img(path, target_size=(64, 64))
    show_img=image.load_img(path, target_size=(200, 200))
    vehicle_class = ['Car', 'Bike']
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x /= 255

    custom = model.predict(x)
    print(custom[0])

    # print(loc[counter])
    ttl = mdl.replace("model_","").replace(".h5","")
    ax = fig.add_subplot(2,2,counter)
    ax.set_title(ttl + " -" +vehicle_class[np.argmax(custom[0])])
    ax.imshow(show_img)
    ax.axis('off')
    counter = counter + 1
    # plt.imshow(show_img)
plt.show()

# a=custom[0]
# ind=np.argmax(a)

# print('Prediction:',vehicle_class[ind])