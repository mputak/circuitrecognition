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

align(arr)
print(arr)
x = [point[0] for point in arr]
y = [point[1] for point in arr]

element_counter = {i: 0 for i in range(1, 10)}
print(element_counter)
# Create the scatter plot
# plt.scatter(x, y)

# # Show the plot
# plt.show()