import numpy as np

# inputs for this neuron which are outputs of neurons from previous layer
inputs = np.array([[1, 2, 3, 2.5],
                   [2.0, 5.0, -1.0, 2.0],
                   [-1.5, 2.7, 3.3, -0.8]])

# each row in weights array is the weights for one neuron in layer
weights = np.array([[0.2, 0.8, -0.5, 1.0],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]])

biases = np.array([2, 3, 0.5])

# layer_outputs = np.zeros(inputs.shape[1])
layer_outputs = []


#                                 broadcasting add
layer_outputs = np.matmul(inputs, weights.T) + biases
print(layer_outputs)
