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
def softmax_function(a) :
    c = np.max(a)
    exp_a = np.exp(a - c)
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
    def printing_out(self, inputs, activation_function):
        output = activation_function(inputs)
        return np.array(output)

class NeuralNetwork:
    def __init__(self, layer_sizes, activation_functions, activation_derivatives):
        self.layers = []
        self.activation_functions = activation_functions
        self.activation_derivatives = activation_derivatives
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i + 1], layer_sizes[i])
            self.layers.append(layer)
    def cross_entropy_error(self, y, t):
        delta = 1e-7
        return -np.sum(t * np.log(y + delta))
    def calculate_output(self, inputs):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            inputs = layer.calculate_output(inputs, self.activation_functions[i])
        inputs = self.activation_functions[-1](inputs)
        return inputs
    def train(self, inputs, targets, learning_rate):
        outputs = self.calculate_output(inputs)
        error = self.cross_entropy_error(outputs, targets)
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

network = NeuralNetwork(layer_sizes=[784, 128, 10],
                        activation_functions=[sigmoid_function, sigmoid_function, softmax_function],
                        activation_derivatives=[lambda x: sigmoid_function(x) * (1 - sigmoid_function(x)),
                                                lambda x: sigmoid_function(x) * (1 - sigmoid_function(x)),
                                                lambda x: sigmoid_function(x) * (1 - sigmoid_function(x))])

def generate_random_array(n):
    """0에서 1사이의 N개 랜덤 원소로 이루어진 배열 생성"""
    arr = [random.random() for _ in range(n)]
    arr = np.array(arr)
    return arr
xxx = generate_random_array(784)
ddd = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 84, 185, 159, 151, 60, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 222, 254, 254, 254, 254, 241, 198, 198, 198, 198, 198, 198, 198, 198, 170, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 114, 72, 114, 163, 227, 254, 225, 254, 254, 254, 250, 229, 254, 254, 140, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 66, 14, 67, 67, 67, 59, 21, 236, 254, 106, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 83, 253, 209, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 233, 255, 83, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 129, 254, 238, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 59, 249, 254, 62, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 133, 254, 187, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 205, 248, 58, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 126, 254, 182, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 75, 251, 240, 57, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 221, 254, 166, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 203, 254, 219, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 254, 254, 77, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 224, 254, 115, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 133, 254, 254, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61, 242, 254, 254, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 254, 254, 219, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 254, 207, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

do = network.calculate_output(ddd)
print(do)