import numpy as np
import random
###공식을 통해 계싼하는 대신 레이어별로 하도록 물론 효율은 나가 뒤졌고, 이해를 위해서.
def activation_function(x):
    return 1 / (1 + np.exp(-x))
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def calculate_output(self, inputs):
        result = []
        for i in range(len(inputs)):
            print(inputs, self.weights, i)
            weighted_input = inputs[i] * self.weights[i]
            result.append(weighted_input)
        weighted_inputs = result
        weighted_sum = sum(weighted_inputs)
        weighted_sum += self.bias
        return activation_function(weighted_sum)

class Layer:
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.neurons = []
        for i in range(number_of_neurons):
            neuron = Neuron(
                weights=[random.uniform(-1, 1) for j in range(number_of_inputs_per_neuron)],
                bias=random.uniform(-1, 1)
            )
            self.neurons.append(neuron)

    def calculate_output(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs
