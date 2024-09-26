# Step function
# ReLU function - Rectified Linear Unit -> most popular activation function for hidden layers in NN - super simple, super fast as simple calculation
# Sigmoid function -> has a problem of vanishing gradient
# Softmax function

# ReLU function implementation
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

for i in inputs:
    output.append(max(0, i))

# print(output)

# the non linear nature (a very slight deviation from linear nature) of the ReLU function is what allows the neural network to fit to non linear data as well
# ReLU function is the most popular activation function for hidden layers in neural networks

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(100, 3)
print(X[:5])
print(len(y))

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Acitvation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = Layer_Dense(2, 5)
activation1 = Acitvation_ReLU()

layer1.forward(X)
activation1.forward(layer1.output)

print(layer1.output)
print(activation1.output)