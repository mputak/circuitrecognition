import torch
import cv2 as cv
import math
import numpy as np
import pandas
import matplotlib.pyplot as plt
import itertools as it

path_to_best = "best_weights.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path_to_best)


class Process:
    def __init__(self, files, threshold_value):
        self.files = files
        self.images = [cv.imread(img, 0) for img in self.files]
        self.threshold_value = threshold_value
        self.valid_wires = []

        self.result = model(self.images, size=640)
        self.df = self.result.pandas().xywh
        self.df_xyxy = self.result.pandas().xyxy
        # self.result.show()
        edge = cv.Canny(self.images[0], 100, 200, L2gradient=True)
        contours, hierarchy = cv.findContours(edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        print(self.df_xyxy)
        cv.imshow("Canny", edge)
        cv.waitKey(0)
        # for (image, df, df_xyxy) in zip(self.images, self.df, self.df_xyxy):
        #     self.trashed = cv.adaptiveThreshold(image, self.threshold_value, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 9)
        #     self.element_masker(image, df_xyxy)
        #     self.line_detector(self.junction_finder(df), image, padding=42)
        #     all_elements = self.element_detector(df)
        # self.write_to_file(all_elements)

    @staticmethod
    def junction_finder(df):
        '''Finds all junctions and outputs them in an array'''
        junctions = df.loc[df['class'] == 4]
        junction_array = junctions[["xcenter", "ycenter"]].to_numpy()

        return junction_array.astype(int)

    def element_masker(self, img, df):
        '''Removes elements from the thresholded image'''
        for row in df.to_numpy():
            if row[5] != 4:
                cv.rectangle(self.trashed, (int(row[0]), int(row[1])), (int(row[2]), int(row[3])), (0, 0, 0), -1)

    def line_detector(self, array, img, padding):
        '''Masks all ROI's and finds all lines'''
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
            lines = (cv.HoughLinesP(masked, rho=1, theta=np.pi / 180, threshold=50, minLineLength=70, maxLineGap=4))
            if np.all(lines):
                self.valid_wires.append(combination)
                # for i in range(0, len(lines)):
                #     l = lines[i][0]
                #     cv.line(self.trashed, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 5, cv.LINE_AA)

        # testing purposes
        cv.imshow("THRASHED AND DETECTED", self.trashed)
        cv.waitKey(0)

    def element_detector(self, df):
        '''Finds all elements and places them on a valid wire'''
        counter = 0
        list_of_elements = []
        for row in df.to_numpy():
            if row[5] != 4:
                dist_to_element = {}
                for wire in self.valid_wires:
                    dx, dy = wire[1][0] - wire[0][0], wire[1][1] - wire[0][1]
                    det = dx * dx + dy * dy
                    a = (dy*(int(row[1]) - wire[0][1]) + dx * (int(row[0]) - wire[0][0])) / det
                    pt = (wire[0][0] + a * dx, wire[0][1] + a * dy)
                    dist_to_element[pt] = math.dist(pt, (row[0], row[1]))
                pt_on_wire = min(dist_to_element, key=dist_to_element.get)
                if row[5] == 5:
                    list_of_elements.append(f"SYMBOL res {round(pt_on_wire[0])} {round(pt_on_wire[1])} R{counter}\nSYMATTR InstName R1\n")
                    counter += 1
        # cv.imshow("element", self.trashed)
        # cv.waitKey(0)
        return list_of_elements

    def write_to_file(self, elements):
        '''Writes all elements and wires to an .asc output file'''
        print(len(elements))
        fo = open("digitized_circuit.asc", "w")
        fo.write("Version 4\nSHEET 1 880 680\n")
        for wire in self.valid_wires:
            fo.write(f"WIRE {wire[0][0]} {wire[0][1]} {wire[1][0]} {wire[1][1]}\n")
        for element in elements:
            print(element)
            fo.write(element)

        fo.close()
