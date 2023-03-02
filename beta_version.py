import torch
import cv2 as cv
import math
import numpy as np
import tkinter as tk
from tkinter.filedialog import askopenfilename

# class that has wires as object with coordinates as attributes
class Wire:
    def __init__(self, x1, y1, x2, y2):
        if self.select_starting_point(x1, y1, x2, y2) == 1:
            self.x_start = x1
            self.y_start = y1
            self.x_end = x2
            self.y_end = y2
        else:
            self.x_start = x2
            self.y_start = y2
            self.x_end = x1
            self.y_end = y1
        # method for defining if the wire is horizontal or vertical
        self.orientation = self.test_if_wire_is_valid()
        # self.straighten_wires()
        # translation
        self.x_trans = None
        self.y_trans = None

    # using the fromula for the slope between two points we get either vertical, horizontal or other
    # limit for the slope can be manually set
    def test_if_wire_is_valid(self):
        slope_as_angle = math.degrees(math.atan2((self.y_start - self.y_end), (self.x_start - self.x_end))) % 360
        limit = 10
        if (not limit < slope_as_angle <= 360 - limit) or (180 - limit <= slope_as_angle < 180 + limit):
            return "h"
        elif (90 - limit < slope_as_angle < 90 + limit) or (270 - limit < slope_as_angle < 270 + limit):
            return "v"
        return "o"
    # method for determining which point of the given wire is closer to the origin, then set it to the starting point
    def select_starting_point(self, x1, y1, x2, y2):
        # point closer to origin
        # origin is (0, 0)
        if np.sqrt(x1**2 + y1**2) < np.sqrt(x2**2 + y2**2):
            return 1
        else:
            return 2
    # failed attempt to straighten the wires
    def straighten_wires(self):
        # straigthens the wire
        if self.orientation == "h":
            self.y_end = self.y_start

        elif self.orientation == "v":
            self.x_end = self.x_start


tk.Tk().withdraw()
print("Choose an image.")
filename = askopenfilename()
# model import
model = torch.hub.load("yolov5-master", 'custom',
                       path="best_2.pt", source='local')  # local repo
img = cv.imread(filename)[..., ::-1]  # image loader
results = model(img, size= 640)  # inference

# basic txt file open for ltspice
fo = open("ltspice_data.asc", "w")
fo.write("Version 4\n")
fo.write("SHEET 1 880 680\n")

results.print()  # print result
results.show()
# results.save()# show or save result

df = results.pandas().xywh[0]
dfxyxy = results.pandas().xyxy[0]
print(df)

# defintion for selecting the starting point of the wire (closer to the origin) (just a copy of a class method)
def select_starting_point(x1, y1, x2, y2):
    # point closer to origin
    # origin is (0, 0)
    if np.sqrt(x1 ** 2 + y1 ** 2) < np.sqrt(x2 ** 2 + y2 ** 2):
        return 1
    else:
        return 2


# ground data processing
data_ind = df.loc[df['name'] == 'inductor']
# ground data processing
data_gnd = df.loc[df['name'] == 'gnd']
# capacitor data processing
data_cap = df.loc[df['name'] == 'capacitor-unpolarized']
# voltage source data processing
data_vol_ac = df.loc[df['name'] == 'voltage-dc_ac']
data_vol = df.loc[df['name'] == 'voltage-dc']
# resistor data processing
data_res = df.loc[df['name'] == 'resistor'] # searching the resistors accross the dataframe
# junction data processing
data = df.loc[df['name'] == 'junction'] # searching for the junctions in the dataframe
# some empty lists neccessary for later itterations
wires = []
checked_coordinates = []
wires_horizontal = []
wires_vertical = []

# itteration of the data and straightening the wires
for index, row in data.iterrows():
    align_x = [] # empty list containing x_coordinates that need to have the same value
    align_y = [] # empty list containing y_coordinates that need to have the same value

    for index2, row2 in data.iterrows():
        if index == index2: # skip the same point
            continue
        if abs(row["xcenter"]-row2["xcenter"]) < 50: # taking two different points and append if they are close
            align_x.append(index) # appending the index of a valid coordinate for easier manipulation of the dataframe
            align_x.append(index2) # appending the index of a second valid coordinate...

        elif abs(row["ycenter"]-row2["ycenter"]) < 50: # taking two different points and
            # if they are close --> append in list
            align_y.append(index) # appending the index of a valid coordinate for easier manipulation of the dataframe
            align_y.append(index2) # appending the index of a second valid coordinate...

    # itterating over the list
    for index in align_x:
        data.loc[index, "xcenter"] = data.loc[align_x[0], "xcenter"] # equating every element of the list to the first
        # in the list
    # itterating over the second list
    for index in align_y:
        data.loc[index, "ycenter"] = data.loc[align_y[0], "ycenter"] # equating every element of the list to the first
        # in the list

# process of translating points to the 16x16 grid for ltspice, x_center
for index, row in data.iterrows():
    if (data.loc[index, "xcenter"]) % 16 == 0:
        continue
    else:
        temp_x = row["xcenter"] % 16
        data.loc[index, "xcenter"] = row["xcenter"] - temp_x

# process of translating points to the 16x16 grid for ltspice, y_center
for index, row in data.iterrows():
    if (data.loc[index, "ycenter"]) % 16 == 0:
        continue
    else:
        temp_y = row["ycenter"] % 16
        data.loc[index, "ycenter"] = row["ycenter"] - temp_y

# itterating over all of the junctions (now aligned junctions)
for index, row in data.iterrows():
    for index2, row2 in data.iterrows():
        if (index, index2) in checked_coordinates or (index2, index) in checked_coordinates: # condition to stop
            # duplication caused by the itteration
            continue
        checked_coordinates.append((index, index2))
        checked_coordinates.append((index2, index))
        # calling the class Wire and making each wire an object
        wire = Wire(
            x1=int(row["xcenter"]),
            y1=int(row["ycenter"]),
            x2=int(row2["xcenter"]),
            y2=int(row2["ycenter"])
        )
        if index == index2:
            continue
        if wire.orientation == "o":
            continue
        elif wire.orientation == "h":
            wires_horizontal.append(wire)
            wires.append(wire) # appending horizontal wires
        elif wire.orientation == "v":
            wires_vertical.append(wire)
            wires.append(wire) # appending vertical wires

counter = 1 # counter for unique names of the elements
# process for placing resistors on the wire with appropriate rotation
for index, row in data_res.iterrows(): # iterate over resistors
    for wire in wires: # iterate over wire object
        if wire in wires_vertical:
            if wire.y_start < row["ycenter"] < wire.y_end and wire.x_start - 35 < row["xcenter"] < wire.x_end + 35:
                # proximity of the resistor to the wire condition
                data_res.loc[index, "xcenter"] = wire.x_start -16
                # let the new x_center be the same as the x_start value of the wire,- 16 is to match the grid
                text1 = "SYMBOL res {} {} R0\nSYMATTR InstName R{}\n"\
                    .format(int(data_res.loc[index, "xcenter"]), int(row["ycenter"]), counter)
                fo.write(text1)
                counter = counter + 1 # counter increment

        elif wire in wires_horizontal:
            if wire.x_start < row["xcenter"] < wire.x_end and wire.y_start - 35 < row["ycenter"] < wire.y_end + 35:
                # proximity of the resistor to the wire condition
                data_res.loc[index, "ycenter"] = wire.y_start - 16
                # let the new y_center be the same as the y_start value of the wire,- 16 is to match the grid
                text2 = "SYMBOL res {} {} R90\nSYMATTR InstName R{}\n"\
                    .format(int(row["xcenter"]), int(data_res.loc[index, "ycenter"]) ,counter)
                fo.write(text2)
                counter = counter + 1 # counter increment
        else:
            continue

counter = 1
for index, row in data_vol.iterrows(): # iterate over voltage-dc sources
    for wire in wires: # itđterate over wire object
        if wire in wires_vertical:
            if wire.y_start < row["ycenter"] < wire.y_end and wire.x_start - 25 < row["xcenter"] < wire.x_end + 25:
                # proximity of the voltage-dc source to the wire
                data_vol.loc[index, "xcenter"] = wire.x_start
                # let the new x_center be the same as the x_start value of the wire,- 16 is to match the grid
                text1 = "SYMBOL voltage {} {} R0\nSYMATTR InstName V{}\n"\
                    .format(int(data_vol.loc[index, "xcenter"]), int(row["ycenter"]) - 96, counter)
                fo.write(text1)
                counter = counter + 1 # counter increment

        elif wire in wires_horizontal:
            if wire.x_start < row["xcenter"] < wire.x_end and wire.y_start - 25 < row["ycenter"] < wire.y_end + 25:
                # proximity of the voltage-dc source to the wire
                data_vol.loc[index, "ycenter"] = wire.y_start
                # let the new y_center be the same as the y_start value of the wire,- 16 is to match the grid
                text2 = "SYMBOL voltage {} {} R270\nSYMATTR InstName V{}\n"\
                    .format( int(row["xcenter"]) - 96, int(data_vol.loc[index, "ycenter"]), counter)
                fo.write(text2)
                counter = counter + 1 # counter increment
        else:
            continue

counter = 1
for index, row in data_vol_ac.iterrows(): # iterate over voltage-dc sources
    for wire in wires: # itđterate over wire object
        if wire in wires_vertical:
            if wire.y_start < row["ycenter"] < wire.y_end and wire.x_start - 25 < row["xcenter"] < wire.x_end + 25:
                # proximity of the voltage-dc source to the wire
                data_vol_ac.loc[index, "xcenter"] = wire.x_start
                text1 = "SYMBOL voltage {} {} R0\nSYMATTR InstName V{}\n"\
                    .format(int(data_vol_ac.loc[index, "xcenter"]), int(row["ycenter"]) - 96, counter)
                fo.write(text1)
                counter = counter + 1 # counter increment

        elif wire in wires_horizontal:
            if wire.x_start < row["xcenter"] < wire.x_end and wire.y_start - 25 < row["ycenter"] < wire.y_end + 25:
                # proximity of the voltage-dc source to the wire
                data_vol_ac.loc[index, "ycenter"] = wire.y_start
                text2 = "SYMBOL voltage {} {} R270\nSYMATTR InstName V{}\n"\
                    .format( int(row["xcenter"]) - 96, int(data_vol_ac.loc[index, "ycenter"]), counter)
                fo.write(text2)
                counter = counter + 1 # counter increment
        else:
            continue

counter = 1
for index, row in data_cap.iterrows(): # iterate over capacitor sources
    for wire in wires: # iterate over wire object
        if wire in wires_vertical:
            if wire.y_start < row["ycenter"] < wire.y_end and wire.x_start - 50 < row["xcenter"] < wire.x_end + 50:
                # proximity of the voltage-dc source to the wire
                data_cap.loc[index, "xcenter"] = wire.x_start - 16
                # let the new x_center be the same as the x_start value of the wire,- 16 is to match the grid
                text1 = "SYMBOL cap {} {} R0\nSYMATTR InstName C{}\n"\
                    .format(int(data_cap.loc[index, "xcenter"]), int(row["ycenter"]), counter)
                fo.write(text1)
                counter = counter + 1 # counter increment

        elif wire in wires_horizontal:
            if wire.x_start < row["xcenter"] < wire.x_end and wire.y_start - 25 < row["ycenter"] < wire.y_end + 25:
                # proximity of the voltage-dc source to the wire
                data_cap.loc[index, "ycenter"] = wire.y_start - 16
                # let the new y_center be the same as the y_start value of the wire,- 16 is to match the grid
                text2 = "SYMBOL cap {} {} R90\nSYMATTR InstName C{}\n"\
                    .format(int(row["xcenter"]), int(data_cap.loc[index, "ycenter"]), counter)
                fo.write(text2)
                counter = counter + 1 # counter increment
        else:
            continue

# process for placing ground on the wire with appropriate rotation
for index, row in data_gnd.iterrows(): # iterate over ground
    for wire in wires: # iterate over wire object
        if wire in wires_horizontal:
            if wire.x_start < row["xcenter"] < wire.x_end and wire.y_start - 200 < row["ycenter"] < wire.y_end + 200:
                # proximity of the resistor to the wire condition
                data_gnd.loc[index, "ycenter"] = wire.y_start + 16
                # let the new y_center be the same as the y_start value of the wire,- 16 is to match the grid
                text2 = "FLAG {} {} 0\n".format(int(row["xcenter"]), int(data_gnd.loc[index, "ycenter"]))
                fo.write(text2)
                text2 = "WIRE {} {} {} {}\n"\
                    .format(int(row["xcenter"]), int(data_gnd.loc[index, "ycenter"] - 16),
                            int(row["xcenter"]), int(data_gnd.loc[index, "ycenter"]))
                fo.write(text2)
        else:
            continue

counter = 1 # counter for unique names of the elements
# process for placing inductors on the wire with appropriate rotation
for index, row in data_ind.iterrows(): # iterate over inductors
    for wire in wires: # iterate over wire object
        if wire in wires_vertical:
            if wire.y_start < row["ycenter"] < wire.y_end and wire.x_start - 35 < row["xcenter"] < wire.x_end + 35:
                # proximity of the resistor to the wire condition
                data_ind.loc[index, "xcenter"] = wire.x_start -16
                # let the new x_center be the same as the x_start value of the wire,- 16 is to match the grid
                text1 = "SYMBOL ind {} {} R0\nSYMATTR InstName L{}\n"\
                    .format(int(data_ind.loc[index, "xcenter"]), int(row["ycenter"]), counter)
                fo.write(text1)
                counter = counter + 1 # counter increment

        elif wire in wires_horizontal:
            if wire.x_start < row["xcenter"] < wire.x_end and wire.y_start - 35 < row["ycenter"] < wire.y_end + 35:
                # proximity of the resistor to the wire condition
                data_ind.loc[index, "ycenter"] = wire.y_start - 16
                # let the new y_center be the same as the y_start value of the wire,- 16 is to match the grid
                text2 = "SYMBOL ind {} {} R90\nSYMATTR InstName L{}\n"\
                    .format(int(row["xcenter"]), int(data_ind.loc[index, "ycenter"]), counter)
                fo.write(text2)
                counter = counter + 1 # counter increment
        else:
            continue

# printing the coordinates in the appropriate form for the ltspice data processing
for wire in wires:
    text = f"WIRE {wire.x_start} {wire.y_start} {wire.x_end} {wire.y_end}\n"
    fo.write(text)
fo.close()

fo = open("ltspice_data.asc")
lines = fo.readlines()

no_duplicates = list()

for object in lines:
    if object not in no_duplicates:
        no_duplicates.append(object)

# New file
final_file = open("ltspice_final.asc", "w")
final_file.writelines(no_duplicates)

fo.close()
final_file.close()
