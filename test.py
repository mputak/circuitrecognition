import torch
import cv2
import math


def test_if_wire_is_valid(x1, y1, x2, y2):
    slope_as_angle = math.degrees(math.atan2((y1 - y2), (x1 - x2))) % 360
    limit = 20
    if (not limit < slope_as_angle <= 360 - limit) or (180 - limit <= slope_as_angle < 180 + limit):
        return "h"
    elif (90 - limit < slope_as_angle < 90 + limit) or (270 - limit < slope_as_angle < 270 + limit):
        return "v"
    return "o"


model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Marko/PycharmProjects/zavrsni/best.pt')  # local model
model = torch.hub.load('C:/Users/Marko/PycharmProjects/zavrsni/yolov5-master', 'custom', path='C:/Users/Marko/PycharmProjects/zavrsni/best.pt', source='local')  # local repo
img = cv2.imread('C:/Users/Marko/Desktop/testv0.jpeg')[..., ::-1]  # image loader

results = model(img, size= 640)  # inference


fo = open("C:/Users/Marko/PycharmProjects/zavrsni/ltspice_data.asc", "w")
fo.write("Version 4\n")
fo.write("SHEET 1 880 680\n")

#results.print()  # print result
#results.show()  # show or save result

#print(results.pandas().xyxy[0])

df = results.pandas().xywh[0]

# resistor data processing
data = df.loc[df['name'] == 'resistor'] # trazo junction po dataframeu
for index, row in data.iterrows():
    text = "SYMBOL res {} {} R90\n".format(int(row["xcenter"]), int(row["ycenter"]))
    fo.write(text)

# junction data processing
data = df.loc[df['name'] == 'junction'] # trazo junction po dataframeu
wires = []
indicies = []
for index, row in data.iterrows():
    for index2, row2 in data.iterrows():
        if index == index2:
            continue
        if test_if_wire_is_valid(x1=row["xcenter"], y1=row["ycenter"],
                                 x2=row2["xcenter"], y2=row2["ycenter"]) == "o":
            continue
        elif test_if_wire_is_valid(x1=row["xcenter"], y1=row["ycenter"],
                                   x2=row2["xcenter"], y2=row2["ycenter"]) == "h":
            wire = []
            #row2["ycenter"] = row["ycenter"]
        elif test_if_wire_is_valid(x1=row["xcenter"], y1=row["ycenter"],
                                   x2=row2["xcenter"], y2=row2["ycenter"]) == "v":
            wire = []
            #row2["xcenter"] = row["xcenter"]

        wire.append(int(row["xcenter"]))
        wire.append(int(row["ycenter"]))
        wire.append(int(row2["xcenter"]))
        wire.append(int(row2["ycenter"]))
        wires.append(wire)

no_duplicate_wires = []
for wire in wires:
    for wire2 in wires:
        if wire[0] == wire2[2] and wire[1] == wire2[3] and wire[2] == wire2[0] and wire[3] == wire2[1] or wire == wire2:
            continue
        else:
            temp = [wire2[2], wire2[3], wire2[0], wire2[1]]
            if wire2 not in no_duplicate_wires and temp not in no_duplicate_wires:
                no_duplicate_wires.append(wire2)

for wire in list(no_duplicate_wires):
    text = f"WIRE {wire[0]} {wire[1]} {wire[2]} {wire[3]}\n"
    fo.write(text)

fo.close()
