# loss function for a neural network with output layer as a softmax activation function is called the Categorical Cross-Entropy Loss function
# Loss = -1 * (summation of (Y * np.log(Prediction)))
# Y - one-hot encoded ground truth label (actual output)
# Prediction - output of the softmax activation function

# the loss function is useful in classification problems

import math

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0]) * target_output[0] +
            math.log(softmax_output[1]) * target_output[1] +
            math.log(softmax_output[2]) * target_output[2])

print(loss)