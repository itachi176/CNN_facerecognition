import cv2
class SimplePreprocessor:
    def __init__(self, width, height, inter = cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocessor(self, image):
        #resize the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)