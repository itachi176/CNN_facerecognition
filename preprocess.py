import numpy as np # pip install numpy
import os
from random import shuffle
from tqdm import tqdm
import cv2
def my_label(image_name):
    name = image_name.split('.')[-3] 
    # if you have two person in your dataset
#     if name=="Ishwar":
#         return np.array([1,0])
#     elif name=="Manish":
#         return np.array([0,1])
    
    
    # if you have three person in your dataset
    if name=="hoang":
        return np.array([1,0,0])
    elif name=="messi":
        return np.array([0,1,0])
    elif name=="ronaldo":
        return np.array([0,0,1])

    
def my_data():
    data = []
    for img in tqdm(os.listdir("./dataset/")):
        path=os.path.join("./dataset/",img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (50,50))
        data.append([np.array(img_data), my_label(img)])
    shuffle(data)  
    return data

# data = my_data()
