# Why batches?
# - To avoid loading the entire dataset into memory
# - helps to perform parallel processing
# - thats why train NN models in gpus instead of cpus
# - helps to generalize the model -> if we show only one sample at a time during training, the neuron will try to fit only that sample. But if we show multiple samples (batch input), the neuron will try to fit all the samples. This will help to generalize the model.

import numpy as np

# 3 sample batch with 4 features
inputs = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

# layer with 3 neurons
weights_layer1 = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

# bias of the 3 neurons
biases_layer1 = [2.0, 3.0, 0.5]

output_layer1 = np.dot(inputs, np.array(weights_layer1).T) + biases_layer1


# layer with 4 neurons
weights_layer2 = [
    [0.1, -0.14, 0.5],
    [-0.5, 0.12, -0.33],
    [-0.44, 0.73, -0.13],
    [0.13, -0.22, 0.17]
]

# bias of the 4 neurons
biases_layer2 = [-1, 2, -0.5, 1]

# output of the first layer is the input of the second layer
output_layer2 = np.dot(output_layer1, np.array(weights_layer2).T) + biases_layer2

print(output_layer2)



############################################################################################################

np.random.seed(0)

X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
layer2.forward(layer1.output)

print(layer2.output)