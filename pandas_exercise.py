import pandas as pd
import matplotlib.pyplot as plt

data = [['tom', 10], ['nick', 15], ['juli', 14]]
  
# Create the pandas DataFrame
df = pd.DataFrame(data, columns=['Name', 'Age'])



# Create the array
arr = [[ 313,  484],
       [ 341,  916],
       [1066,  847],
       [ 866,  874],
       [ 596,  485],
       [1027,  492],
       [ 820,  491],
       [ 615,  896]]


def equate_points(points):
    # Create dictionaries to store points that are horizontally and vertically aligned
    horizontal_points = {}
    vertical_points = {}

    # Iterate through the input points and group them based on their horizontal and vertical alignment
    for point in points:
        x, y = point

        # Check if the point is horizontally aligned with any other points
        for other_x, other_y in horizontal_points:
            if abs(x - other_x) <= 50:
                horizontal_points[other_x, other_y].append(point)
                break
        else:
            horizontal_points[x, y] = [point]

        # Check if the point is vertically aligned with any other points
        for other_x, other_y in vertical_points:
            if abs(y - other_y) <= 50:
                vertical_points[other_x].append(point)
                break
        else:
            vertical_points[x] = [point]

    # Create a new list to store the equated points
    equated_points = []

    # Iterate through the horizontally aligned points and equate their x-coordinates
    for x, y_list in horizontal_points.items():
        min_x = min(point[0] for point in y_list)
        for point in y_list:
            equated_points.append((min_x, point[1]))

    # Iterate through the vertically aligned points and equate their y-coordinates
    for x, y_list in vertical_points.items():
        min_y = min(point[1] for point in y_list)
        for point in y_list:
            equated_points.append((point[0], min_y))

    # Add any remaining unaligned points to the equated points list
    for point in points:
        if point not in equated_points:
            equated_points.append(point)

    return equated_points

# Extract the x and y coordinates as separate arrays
def align(arr):
    '''
    Input: array
    Return: aligned_array
    '''
    for i, val in enumerate(arr):
        for j, val2 in enumerate(arr):
            if i == j:
                continue
            elif abs(val[0] - val2[0]) < 50:
                arr[j][0] = val[0]
            elif abs(val[1] - val2[1]) < 50:
                arr[j][1] = val[1]
    return arr

# align(arr)
# print(arr)


# element_counter = {i: 0 for i in range(1, 10)}
# print(element_counter)
# Create the scatter plot
eq_p = equate_points(arr)
x = [point[0] for point in eq_p]
y = [point[1] for point in eq_p]
plt.scatter(x, y)

# Show the plot
plt.show()