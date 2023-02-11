import neuron
Layer = neuron.Layer
Neuron = neuron.Neuron
sigmoid_function =neuron.sigmoid_function
relu_function = neuron.relu_function
softmax_function = neuron.softmax_function

# Create the input layer
input_layer = Layer(3, 3)

# Create the hidden layer
hidden_layer = Layer(4, 3)

# Create the output layer
output_layer = Layer(3, 4)

# Define the inputs to the network
inputs = [1, 2, 3]

# Perform a forward pass through the input layer
input_layer_output = input_layer.calculate_output(inputs, sigmoid_function)

# Perform a forward pass through the hidden layer
hidden_layer_output = hidden_layer.calculate_output(input_layer_output, sigmoid_function)

# Perform a forward pass through the output layer
outputs = output_layer.calculate_output(hidden_layer_output, softmax_function)


print(hidden_layer_output)
print("Outputs:", outputs)
