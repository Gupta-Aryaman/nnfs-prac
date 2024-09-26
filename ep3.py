l = [1, 2, 3] # array -> 1D list -> shape = (3,)

lol = [
    [1, 2, 3],
    [4, 5, 6]    
] # 2D list -> shape = (2, 3)

lolol = [
    [
        [1, 2, 3, 4],
        [4, 5, 6, 7]
    ],
    [
        [7, 8, 9, 10],
        [10, 11, 12, 13]
    ],
    [
        [13, 14, 15, 16],
        [16, 17, 18, 19]
    ]
] # 3D list -> shape = (3, 2, 4)

# A tensor is an object can be represented as an array


# Dot product using numpy
import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

output = np.dot(weights, inputs) + bias
print(output)

input_for_2_neurons = [1.0, 2.0, 3.0]
bias_for_2_neurons = [2.0, 3.0]
output = np.dot(lol, input_for_2_neurons) + bias_for_2_neurons
print(output)