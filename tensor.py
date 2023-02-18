import numpy as np
import random
import tensorflow as tf
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()



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
            print(i, len(inputs))
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
        
        # Update the weights and biases of the neurons in the network
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            for j in range(len(layer.neurons)):
                neuron = layer.neurons[j]
                for k in range(len(neuron.weights)):
                    neuron.weights[k] -= delta * inputs[k]
                neuron.bias -= delta
            
            # Calculate the error for the next layer
            if i > 0:
                inputs = [neuron.calculate_output(inputs, self.activation_functions[i - 1]) for neuron in layer.neurons]
                delta = derivative * self.activation_derivatives[i - 1](inputs) * learning_rate




network = NeuralNetwork(layer_sizes=[784, 784, 10],
                        activation_functions=[sigmoid_function, sigmoid_function, softmax_function],
                        activation_derivatives=[lambda x: sigmoid_function(x) * (1 - sigmoid_function(x)),
                                                lambda x: sigmoid_function(x) * (1 - sigmoid_function(x)),
                                                lambda x: sigmoid_function(x) * (1 - sigmoid_function(x))])

x_train_flat = x_train.reshape(x_train.shape[0], 784)
x_test_flat = x_test.reshape(x_test.shape[0], 784)


def gen_target_arr(n):
    arr = np.array([0,0,0,0,0,0,0,0,0,0])
    arr[n] = 1
    return arr


# Train the network
for i in range(100):
    for j in range(len(x_train_flat)):
        network.train(x_train_flat[j], gen_target_arr(y_train[j]) , 0.1)



"""
correct = 0
for i in range(len(x_test)):
    output = network.calculate_output(x_test[i])
    if np.argmax(output) == y_test[i]:
        correct += 1

print("Accuracy:", correct / len(x_test))
"""
