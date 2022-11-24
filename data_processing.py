import torch
import cv2 as cv
import math
import numpy as np
import pandas


path_to_best = "/home/marko/PycharmProjects/circuitrecognition/best_2.pt"

model = torch.hub.load("ultralytics/yolov5", "custom", path_to_best)
'''
img = cv.imread(GUI.root.files)[..., ::-1]

'''


class Process:
    def __init__(self, files):
        self.files = files
        self.images = [cv.imread(img, 0) for img in self.files]
        self.result = model(self.images, size=640)
        self.df = self.result.pandas().xywh
        print(self.df)
        self.result.show()

