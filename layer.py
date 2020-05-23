import numpy as np

# inputs for this neuron which are outputs of neurons from previous layer
inputs = np.array([1, 2, 3, 2.5])
weights = np.array([[0.2, 0.8, -0.5, 1.0],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]])

biases = np.array([2, 3, 0.5])

layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    neuron_output = np.dot(inputs, neuron_weights) + neuron_bias
    layer_outputs.append(neuron_output)
print(layer_outputs)
