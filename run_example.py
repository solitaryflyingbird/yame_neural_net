import neuron
Layer = neuron.Layer
Neuron = neuron.Neuron

# Create the input layer
input_layer = Layer(3, 3)

# Create the hidden layer
hidden_layer = Layer(4, 3)

# Create the output layer
output_layer = Layer(2, 4)

# Define the inputs to the network
inputs = [1, 2, 3]

# Perform a forward pass through the input layer
input_layer_output = input_layer.calculate_output(inputs)

# Perform a forward pass through the hidden layer
hidden_layer_output = hidden_layer.calculate_output(input_layer_output)

# Perform a forward pass through the output layer
outputs = output_layer.calculate_output(hidden_layer_output)

print("Outputs:", outputs)
