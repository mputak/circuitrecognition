import torch
import cv2 as cv
import math
import numpy as np
import pandas

# make this a parameter to Process class?
# Keep the code clean.
path_to_best = "/home/marko/PycharmProjects/circuitrecognition/best_2.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path_to_best)


class Process:
    def __init__(self, files):
        self.files = files
        self.images = [cv.imread(img, 0) for img in self.files]
        self.result = model(self.images, size=640)
        self.df = self.result.pandas().xywh
        self.line_detector()
        # print(self.df)
        # self.result.show()

    # Method for using linedetector on the grayscale self.images list.

    # FIX!: - it needs to be able to input and output batches of images
    #       - it's not detecting lines as it should but the general idea is here.

    def line_detector(self):
        canny_edge = [cv.Canny(img, 100, 200) for img in self.images]
        cv.imshow("CANNY", canny_edge[0])
        lines = cv.HoughLinesP(canny_edge[0], 1, np.pi / 180, 50, None, 50, 10)

        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv.line(self.images[0], (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

        cv.imshow("sussy", self.images[0])
