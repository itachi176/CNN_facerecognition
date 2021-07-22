from  preprocess import *
import numpy as np
import cv2
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import pickle
data = my_data()

train = data[:400]  
test = data[400:]
X_train = np.array([i[0] for i in train]).reshape(-1,50,50,1)
print(X_train.shape)
y_train = [i[1] for i in train]
X_test = np.array([i[0] for i in test]).reshape(-1,50,50,1)
print(X_test.shape)
y_test = [i[1] for i in test]

from tensorflow.python.framework import ops
ops.reset_default_graph()
convnet = input_data(shape=[50,50,1])
convnet = conv_2d(convnet, 32, 5, activation='relu')
# 32 filters and stride=5 so that the filter will move 5 pixel or unit at a time
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 3, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate = 0.001, loss='categorical_crossentropy')
model = tflearn.DNN(convnet, tensorboard_verbose=1)
model.fit(X_train, y_train, n_epoch=12, validation_set=(X_test, y_test), show_metric = True, run_id="FRS" )


# def data_for_visualization():
#     Vdata = []
#     for img in tqdm(os.listdir("test")):
#         path = os.path.join("test", img)
#         img_num = img.split('.')[0] 
#         print("img num:", img_num)
#         img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         img_data = cv2.resize(img_data, (50,50))
#         Vdata.append([np.array(img_data), img_num])
#     shuffle(Vdata)
#     return Vdata

# Vdata = data_for_visualization()

# import matplotlib.pyplot as plt   # pip install matplotlib

# fig = plt.figure(figsize=(20,20))
# for num, data in enumerate(Vdata[:20]):
#     img_data = data[0]
#     y = fig.add_subplot(5,5, num+1)
#     image = img_data
#     print('img_data:', img_data.shape)
#     data = img_data.reshape(50,50,1)
#     model_out = model.predict([data])[0]
    
    # if np.argmax(model_out) == 0:
    #     my_label = 'hoang'
    # elif np.argmax(model_out) == 1:
    #     my_label = 'messi'
    # else:
    #     my_label = 'ronaldo'
        
#     y.imshow(image, cmap='gray')
#     plt.title(my_label)
    
#     y.axes.get_xaxis().set_visible(False)
#     y.axes.get_yaxis().set_visible(False)
# print (data)
# plt.show()
img = cv2.imread("./test/hoang.1.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (50,50))
data = []
data.append([np.array(img), 'hoang'])
img = img.reshape(50,50,1)
x = model.predict([img])
if np.argmax(x) == 0:
    my_label = 'hoang'
elif np.argmax(x) == 1:
    my_label = 'messi'
else:
    my_label = 'ronaldo'
        
print(my_label)