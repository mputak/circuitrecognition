import torch
import cv2 as cv
import math
import numpy as np


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

        self.orientation = self.test_if_wire_is_valid()
        # self.straighten_wires()
        # translation
        self.x_trans = None
        self.y_trans = None
        # neighbors
        self.neighbors = []
        self.neighbors_by_start = []
        self.neighbors_by_end = []

    def test_if_wire_is_valid(self):
        slope_as_angle = math.degrees(math.atan2((self.y_start - self.y_end), (self.x_start - self.x_end))) % 360
        limit = 15
        if (not limit < slope_as_angle <= 360 - limit) or (180 - limit <= slope_as_angle < 180 + limit):
            return "h"
        elif (90 - limit < slope_as_angle < 90 + limit) or (270 - limit < slope_as_angle < 270 + limit):
            return "v"
        return "o"

    def select_starting_point(self, x1, y1, x2, y2):
        # point closer to origin
        # origin is (0, 0)
        if np.sqrt(x1**2 + y1**2) < np.sqrt(x2**2 + y2**2):
            return 1
        else:
            return 2

    def straighten_wires(self):
        # straigthens the wire
        if self.orientation == "h":
            self.y_end = self.y_start

        elif self.orientation == "v":
            self.x_end = self.x_start





#def edge_alignment(x1, y1, x2, y2):
 #   if x1 in range(x2-100, x2+100, 1) or y1 in range(y2-100, y2+100, 1):
  #      x1 = x2
   #     y1 = y2
    #else:
     #   print('Nisu blizu.')

    #return x1, y1, x2, y2


model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Marko/PycharmProjects/zavrsni/best.pt')  # local model
model = torch.hub.load('C:/Users/Marko/PycharmProjects/zavrsni/yolov5-master', 'custom', path='C:/Users/Marko/PycharmProjects/zavrsni/best.pt', source='local')  # local repo
img = cv.imread('C:/Users/Marko/Desktop/testv0.jpeg')[..., ::-1]  # image loader
results = model(img, size= 640)  # inference


fo = open("C:/Users/Marko/PycharmProjects/zavrsni/ltspice_data.asc", "w")
fo.write("Version 4\n")
fo.write("SHEET 1 880 680\n")

#results.print()  # print result
#results.show()  # show or save result

#print(results.pandas().xyxy[0])

df = results.pandas().xywh[0]
dfxyxy = results.pandas().xyxy[0]

# resistor data processing
data = df.loc[df['name'] == 'resistor'] # trazi resistore po dataframeu
for index, row in data.iterrows():
    text = "SYMBOL res {} {} R90\n".format(int(row["xcenter"]), int(row["ycenter"]))
    fo.write(text)



def select_starting_point(x1, y1, x2, y2):
    # point closer to origin
    # origin is (0, 0)
    if np.sqrt(x1 ** 2 + y1 ** 2) < np.sqrt(x2 ** 2 + y2 ** 2):
        return 1
    else:
        return 2

# junction data processing
data = df.loc[df['name'] == 'junction'] # trazi junction po dataframeu
wires = []
checked_coordinates = []
wires_horizontal = []
wires_vertical = []


for index, row in data.iterrows():
    align_x = []
    align_y = []

    for index2, row2 in data.iterrows():
        if index == index2:
            continue
        if abs(row["xcenter"]-row2["xcenter"]) < 50:
            align_x.append(index)
            align_x.append(index2)
            #start = select_starting_point(row["xcenter"], row["ycenter"], row2["xcenter"], row2["ycenter"])
            #if start == 1:
             #   data.loc[index2, "xcenter"] = row["xcenter"]
                #row2["xcenter"] = row["xcenter"]
            #else:
             #   data.loc[index, "xcenter"] = row2["xcenter"]
                #row["xcenter"] = row2["xcenter"]
        elif abs(row["ycenter"]-row2["ycenter"]) < 50:
            align_y.append(index)
            align_y.append(index2)
            #start = select_starting_point(row["xcenter"], row["ycenter"], row2["xcenter"], row2["ycenter"])
            #if start == 1:
             #   data.loc[index2, "ycenter"] = row2["ycenter"]
                #row2["ycenter"] = row["ycenter"]
            #else:
             #   data.loc[index, "ycenter"] = row2["ycenter"]
                #row["ycenter"] = row2["ycenter"]
    for index in align_x:
        data.loc[index, "xcenter"] = data.loc[align_x[0], "xcenter"]

    for index in align_y:
        data.loc[index, "ycenter"] = data.loc[align_y[0], "ycenter"]
for index, row in data.iterrows():
    for index2, row2 in data.iterrows():
        if (index, index2) in checked_coordinates or (index2, index) in checked_coordinates:
            continue
        checked_coordinates.append((index, index2))
        checked_coordinates.append((index2, index))
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
            wires.append(wire)
        elif wire.orientation == "v":
            wires_vertical.append(wire)
            wires.append(wire)



for wire in wires:
    for wire2 in wires:
        if wire.x_start == wire2.x_start and wire.y_start == wire2.y_start and wire.x_end == wire2.x_end and wire.y_end == wire2.y_end:
            continue
        if wire.x_start == wire2.x_start and wire.y_start == wire2.y_start or \
           wire.x_end == wire2.x_end and wire.y_end == wire2.y_end or \
           wire.x_start == wire2.x_end and wire.y_start == wire2.y_end or \
           wire.x_end == wire2.x_start and wire.y_end == wire2.y_start:
            # append neighbor to wire
            wire.neighbors.append(wire2)
'''
for wire in wire.neighbors:
    for wire2 in wire.neighbors:
        if self.orientation == "h":
            if self.y_end != self.y_start:

        elif self.orientation == "v":
            if self.x_end != self.x_start:
'''
#for wire in wire.neighbors:
    #for wire2 in wire.neighbors:
       # wire2.straighten_wires()

    #if wire.x_start == wire2.x_start and wire.y_start == wire2.y_start:
        #    wire.neighbors_by_start.append(wire2)
        #elif wire.x_end == wire2.x_end and wire.y_end == wire2.y_end:
        #    wire.neighbors_by_end.append(wire2)

'''
straight_wires = []
no_duplicate_wires = []
for wire in wires:
    for wire2 in wires:
        if wire[0] == wire2[2] and wire[1] == wire2[3] and wire[2] == wire2[0] and wire[3] == wire2[1] or wire == wire2:
            continue
        else:
            temp = [wire2[2], wire2[3], wire2[0], wire2[1]]
            if wire2 not in no_duplicate_wires and temp not in no_duplicate_wires:
                no_duplicate_wires.append(wire2)
'''

'''
data_bbox = dfxyxy.loc[df['name'] == 'junction']
for wire in straight_wires:
    for wire2 in straight_wires:
        for index, row in data_bbox.iterrows():
        ## nonsense if row["xmin"] < wire[0] and wire[2] < row["xmax"] and row["ymin"] < wire[1] and wire[3] < row["ymax"] and row["xmin"] < wire2[0] and wire2[2] < row["xmax"] and row["ymin"] < wire2[1] and wire2[3] < row["ymax"]: ### joj mene joj
            if row["xmin"] < wire[0] < row["xmax"] and row["xmin"] < wire2[2] < row["xmax"] and row["ymin"] < wire[1] < row["ymax"] and row["ymin"] < wire2[3] < row["ymax"]:
                wire[0] = wire2[2]
                wire[1] = wire2[3] # almost progressing!?
            else:
                continue

'''

for wire in wires:
    text = f"WIRE {wire.x_start} {wire.y_start} {wire.x_end} {wire.y_end}\n"
    fo.write(text)
fo.close()
'''
count = 0
count2 = 1

for wire in straight_wires:
    if (straight_wires[count][2]- straight_wires[count2][0]) < 100 and (straight_wires[count][3]- straight_wires[count2][1]) < 100:
        straight_wires[count][2] = straight_wires[count2][0]
        straight_wires[count][3] = straight_wires[count2][1]
        count = count + 1
        count2 = count2 + 1
        print('yes')
        wire[2] = straight_wires[count][0]
        wire[3] = straight_wires[count2][1]

    elif count2 >= 8:
        break
    else:
        count = count + 1
        count2 = count2 + 1
        print("Nisu blizu cnt1 {} + cnt2 {}".format(count, count2))
for wire in straight_wires:
    text = f"WIRE {straight_wires[0]} {straight_wires[1]} {straight_wires[2]} {straight_wires[3]}\n"
    fo.write(text)
    '''


