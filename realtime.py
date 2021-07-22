import cv2
import mtcnn
import pickle
# face_detector = mtcnn.MTCNN()
# vc = cv2.VideoCapture(0)
# conf_t = 0.99

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

train = data 
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

# while vc.isOpened():
#     ret, frame = vc.read()
#     if not ret:
#         print(':(')
#         break
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_detector.detect_faces(frame_rgb)
#     for res in results:
#         x1, y1, width, height = res['box']
#         x1, y1 = abs(x1), abs(y1)
#         x2, y2 = x1 + width, y1 + height

#         confidence = res['confidence']
#         if confidence < conf_t:
#             continue
#         # key_points = res['keypoints'].values()
#         new = frame[y1:y2, x1:x2]
#         new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
#         new = cv2.resize(new, (50,50))
#         new = new.reshape(50,50,1)
#         result = model.predict([new])[0]
#         if np.argmax(result) == 0:
#             print('hoang')
        
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
#         cv2.putText(frame, f'conf: {confidence:.3f}', (x1, y1), cv2.FONT_ITALIC, 1, (0, 0, 255), 1)

#         # for point in key_points:
#         #     cv2.circle(frame, point, 5, (0, 255, 0), thickness=-1)

#     cv2.imshow('friends', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

temp = 0
while(True):
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(frame, 1.3, 5)
    for x,y,w,h in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h),(255,0,0), 1)
        new = frame[y:y+h, x:x+w]
        new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
        new = cv2.resize(new, (50,50))
        new = new.reshape(50,50,1)
        result = model.predict([new])[0]
        if np.argmax(result) == 0:
            my_label = 'hoang'
        cv2.putText(frame, '{}'.format(my_label), (x, y), cv2.FONT_ITALIC, 1, (0, 0, 255), 1)

        
    cv2.imshow("frame", frame)
    if cv2.waitKey(100) & 0xFF==ord('q'):
        break

cv2.destroyAllWindows()