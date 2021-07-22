import  cv2 
import os 
import numpy as np 

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        if self.preprocessors is None:
            self.preprocessors=[]

    def load(self, imagePaths, verbose = -1):
        data = []
        labels = []
        datas = []
        code = 0
        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            
            if label =='hoang':
                code = 0
            if label =='messi':
                code = 1
            if label == 'ronaldo':
                code = 2
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocessor(image)

            datas.append([np.array(data), code])

            if verbose>0 and i >1 and (i+1)%verbose ==0:
                print("[INFO] process: {}/{}".format(i+1, len(imagePaths)))

        return datas