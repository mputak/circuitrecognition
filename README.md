# circuitrecognition
Go to master branch. Main working areas are main.py and data_processing.py
Proposed valid_wire fix: connect junctions to ends of nearest elements (terminal)
- find how many lines go from certain junction and connect to that amount of nearest terminal points.
- to find the lines from junction see how many lines intersect bounding box of each junction.
- to find terminal points see where bounding box of element bounding box intersects a line.
- append all terminal points to a list so for each junction min_dist can be calculated for all terminal points.
note: this solution does not exclude diagonal direction of wires. 

- Look into appropriate element rotation through slope of a line through 2 points.
