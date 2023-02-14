import numpy as np
import random
###공식을 통해 계싼하는 대신 레이어별로 하도록 물론 효율은 나가 뒤졌고, 이해를 위해서.
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def relu_function(x):
	return np.maximum(0, x)

def softmax_function(x) :
    exp_a = np.exp(x)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a   
    return y

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def calculate_output(self, inputs, activation_function):
        result = []
        for i in range(len(inputs)):
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

    def calculate_output(self, inputs, activation_function):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs, activation_function))
        return outputs

class NeuralNetwork:
    def __init__(self, layer_sizes, activation_functions, activation_derivatives):
        self.layers = []
        self.activation_functions = activation_functions
        self.activation_derivatives = activation_derivatives
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i + 1], layer_sizes[i])
            self.layers.append(layer)

    def calculate_output(self, inputs):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            inputs = layer.calculate_output(inputs, self.activation_functions[i])
        return inputs
    def train(self, inputs, targets, learning_rate):
        outputs = self.calculate_output(inputs)
        error = targets - outputs
        derivative = error * self.activation_derivatives[-1](outputs)
        delta = derivative * learning_rate
        for i in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[i]
            error = np.dot(delta, layer.neurons[0].weights)
            derivative = error * self.activation_derivatives[i - 1](layer.calculate_output(inputs, self.activation_functions[i - 1]))
            delta = derivative * learning_rate
            for j in range(len(layer.neurons)):
                neuron = layer.neurons[j]
                neuron.weights += learning_rate * delta[j] * inputs
                neuron.bias += learning_rate * delta[j]
            inputs = layer.calculate_output(inputs, self.activation_functions[i - 1])


network = NeuralNetwork(layer_sizes=[2, 3, 1],
                        activation_functions=[sigmoid_function, sigmoid_function, sigmoid_function],
                        activation_derivatives=[lambda x: sigmoid_function(x) * (1 - sigmoid_function(x)),
                                                lambda x: sigmoid_function(x) * (1 - sigmoid_function(x)),
                                                lambda x: sigmoid_function(x) * (1 - sigmoid_function(x))])
print(network.calculate_output([1,5]))