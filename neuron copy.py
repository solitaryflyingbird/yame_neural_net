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
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
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
    #예측
    def calculate_output(self, inputs):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            inputs = layer.calculate_output(inputs, self.activation_functions[i])
        inputs = self.activation_functions[-1](inputs)
        return inputs
    #손실함수
    def loss(self, x, t):
        return cross_entropy_error(x, t)
    def train(self, inputs, targets, learning_rate):
        outputs = self.calculate_output(inputs)
        error = self.loss(outputs, targets)
        print(error)
        for i in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[i]
            for j in range(len(layer.neurons)):
                error = self.loss(outputs, targets)
                neuron = layer.neurons[j]
                for k in range(len(neuron.weights)):
                    neuron.weights[k]+=learning_rate
                outputs = self.calculate_output(inputs)
                error2 = self.loss(outputs, targets)
                if error2>error:
                    for k in range(len(neuron.weights)):
                        neuron.weights[k]-=learning_rate*2
                    outputs = self.calculate_output(inputs)
                    error2 = self.loss(outputs, targets)
                    if error2>error:
                        for k in range(len(neuron.weights)):
                            neuron.weights[k]-=learning_rate*2
                    else:
                        error = error2
                else:
                    error = error2
                print(error)

            





network = NeuralNetwork(layer_sizes=[10, 10, 2],
                        activation_functions=[sigmoid_function, sigmoid_function, softmax_function],
                        activation_derivatives=[lambda x: sigmoid_function(x) * (1 - sigmoid_function(x)),
                                                lambda x: sigmoid_function(x) * (1 - sigmoid_function(x)),
                                                lambda x: sigmoid_function(x) * (1 - sigmoid_function(x))])

def decimal_to_binary(decimal):
    ##decimal is an integer less than 2047.
    xx = decimal
    binary = ""
    while decimal > 0:
        binary = str(decimal % 2) + binary
        decimal = decimal // 2
    if len(binary)<10:
        binary = (10-len(binary))*"0" + binary
    ans1 = list(map(int, binary))
    ans2 = xx%2
    return (ans1, ans2)

def separate_tuples(arr):
    arr1 = []
    arr2 = []
    for tup in arr:
        arr1.append(tup[0])
        arr2.append(tup[1])
    return (arr1, arr2)
def swap_elements(arr):
    new_arr = []
    for element in arr:
        if element == 1:
            new_arr.append((1, 0))
        else:
            new_arr.append((0, 1))
    return new_arr

random_nums = np.random.randint(low=0, high=1024, size=200)
binary_nums = [decimal_to_binary(num) for num in random_nums]
input_data, output_data,  = separate_tuples(binary_nums)
output_data = swap_elements(output_data)

x =network.calculate_output(input_data[0])


for i in range(200):
    network.train(input_data[i], output_data[i] , 0.05)
