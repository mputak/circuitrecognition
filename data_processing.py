import torch
import cv2 as cv
import math
import numpy as np
import pandas
import matplotlib.pyplot as plt
import itertools as it

path_to_best = "best_2.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path_to_best)


class Process:
    def __init__(self, files, threshold_value):
        self.files = files
        self.images = [cv.imread(img, 0) for img in self.files]
        self.threshold_value = threshold_value
        self.valid_wires = []

        self.result = model(self.images, size=640)
        self.df = self.result.pandas().xywh  # removed [0] to have whole df in place
        self.df_xyxy = self.result.pandas().xyxy

        for (image, df, df_xyxy) in zip(self.images, self.df, self.df_xyxy):  # added xyxy format for masking elements
            ret, self.trashed = cv.threshold(image, self.threshold_value, 255, cv.THRESH_BINARY_INV)
            self.element_masker(image, df_xyxy)
            # self.line_detector(self.junction_finder(df), image)

        # self.result.show()

    @staticmethod
    def junction_finder(df):  # new argument and made method static (please change that)
        '''Finds all junctions and outputs them in an array'''
        junctions = df.loc[df['class'] == 4]  # removed self.df from both df in a line
        junction_array = junctions[["xcenter", "ycenter"]].to_numpy()

        return junction_array.astype(int)

    def element_masker(self, img, df): # new method for deleting elements

        elements_array = df.to_numpy()
        print(elements_array)
        # all_elements = df.loc[df['class'] != 4]
        mask = np.zeros(img.shape, dtype="uint8")
        # array_of_elements = all_elements.to_numpy()

        for row in elements_array:  # FIX!: make integers of floats
            if row[5] != 4:
                cv.rectangle(mask, (int(row[0]), int(row[1])), (int(row[2]), int(row[3])), (255,255,255), -1) #int dont work
        cv.imshow("Maska ones", mask)

    def line_detector(self, array, img):  # I am very certain OOP doesn't work like that
        '''Masks all ROI's and finds all lines'''
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

            mask = np.zeros(img.shape, dtype="uint8")
            cv.fillPoly(mask, [box], 255)
            masked = cv.bitwise_and(self.trashed, self.trashed, mask=mask)
            lines = (cv.HoughLinesP(masked, rho=1, theta=np.pi / 180, threshold=50, minLineLength=50, maxLineGap=1))
        # indent if under this to be in for loop to draw every line detection
            if lines is not None:
                # new_empty_list_of_valid_wires.append(combination) --> after all iterations, return this list
                self.valid_wires.append(combination)
                for i in range(0, len(lines)):
                    l = lines[i][0]
                    cv.line(self.trashed, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 5, cv.LINE_AA)
                    
        # return self.valid_wires
        # testing purposes
        cv.imshow("THRASHED AND DETECTED", self.trashed)
        cv.imshow("BITWISE", masked)
        cv.waitKey(0)

    def nothing(self):
        pass

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