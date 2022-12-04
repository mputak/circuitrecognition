import skimage.io
import torch
import cv2 as cv
import math
import numpy as np
import pandas
from skimage import morphology, filters
import matplotlib.pyplot as plt
import itertools as it

path_to_best = "best_2.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path_to_best)


class Process:
    def __init__(self, files):
        self.files = files
        self.images = [cv.imread(img, 0) for img in self.files]
        self.mask = np.zeros(self.images[0].shape, dtype="uint8")  # probably unnecessary

        self.result = model(self.images, size=640)
        self.df = self.result.pandas().xywh[0]

        # self.trashed = cv.threshold(self.images[0], cv.COLOR_BGR2GRAY, 200, 255, cv.THRESH_BINARY)
        self.mask_maker(self.junction_finder())
        # self.line_detector()
        print(self.df)
        # self.result.show()

    # FIX!: - it needs to be able to input and output batches of images

    def junction_finder(self):
        '''Finds all junctions and outputs them in an array'''

        junctions = self.df.loc[self.df['class'] == 4]
        junction_array = junctions[["xcenter", "ycenter"]].to_numpy()

        return junction_array.astype(int)

    def mask_maker(self, array):
        '''Masks all ROI's'''

        # lines = []
        padding = 42  # padding for the ROI, 42 seems to work well as it's the number of the gods.
        junction_combinations = list(it.combinations(array, 2))

        for combination in junction_combinations:
            rect = cv.minAreaRect(np.asarray(combination))
            (center, (w, h), angle) = rect
            if w <= h:
                w += padding
            else:
                h += padding
            rect = (center, (w, h), angle)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            mask = np.zeros(self.images[0].shape, dtype="uint8")
            cv.fillPoly(mask, [box], 255)  # was self.mask
            masked = cv.bitwise_and(self.images[0], self.images[0], mask=mask)  # Just change here to self.trashed!!!
            lines = (cv.HoughLinesP(masked, 1, np.pi / 180, 50, None, 0, 0))
            # cv.drawContours(self.images[0], [box], 0, (0, 255, 255), 2)
        # cv.drawContours(self.images[0], [box], 0, (0, 0, 0), 20)

        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv.line(self.images[0], (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

        # to fix the issue of HoughLinesP doing nothing good, prior to calling it, make an inverse threshold function,
        # to leave only the possible "line" part in white, and the rest of the function is black
        # see if you can do the threshold once or do you need to do it every iteration.
        cv.imshow("Output from HoughLinesP", self.images[0])
        cv.imshow("MASK", self.mask)  # unnecessary
        cv.imshow("BITWISE", masked)
        cv.waitKey(0)

    def line_detector(self):
        canny_edge = [cv.Canny(img, 100, 200) for img in self.images]
        # cv.imshow("CANNY", canny_edge[0])
        # (T, threshInv) = cv.threshold(self.images[0], 0, 255,
        #                               cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

        # thresh = cv.adaptiveThreshold(self.images[0], 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 19, 9)


        '''
        lines = cv.HoughLinesP(canny_edge[0], 1, np.pi / 180, 50, None, 50, 10)

        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv.line(self.images[0], (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

        cv.imshow("sussy", self.images[0])
'''