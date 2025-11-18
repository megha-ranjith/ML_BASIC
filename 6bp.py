import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

num_input_nodes = int(input("Enter the number of input nodes: "))
num_hidden_layers = int(input("Enter the number of hidden layers: "))

hidden_layer_sizes = []
for i in range(num_hidden_layers):
    size = int(input(f"Enter the number of nodes in hidden layer {i+1}: "))
    hidden_layer_sizes.append(size)

num_output_nodes = int(input("Enter the number of output nodes: "))

print("\nNetwork Structure:")
print(f"Number of input nodes: {num_input_nodes}")
print(f"Number of hidden layers: {num_hidden_layers}")
print(f"Number of nodes in each hidden layer: {hidden_layer_sizes}")
print(f"Number of output nodes: {num_output_nodes}\n")

inputs = np.array([float(input(f"Enter the value of x{i+1}: ")) for i in range(num_input_nodes)])
target = np.array([float(input("Enter the target output (y): "))])
learning_rate = float(input("Enter the learning rate: "))


layer_sizes = [num_input_nodes] + hidden_layer_sizes + [num_output_nodes]

weights = []
biases = []
for i in range(len(layer_sizes)-1):
    weights.append(np.random.uniform(-1, 1, (layer_sizes[i], layer_sizes[i+1])))
    biases.append(np.random.uniform(-1, 1, (layer_sizes[i+1],)))

#forward pass function
def forward_pass(x):
    activations = [x]
    zs = []  # weighted sums
    for w, b in zip(weights, biases):
        z = np.dot(activations[-1], w) + b
        zs.append(z)
        a = sigmoid(z)
        activations.append(a)
    return activations, zs

#\Backward pass function
def backward_pass(activations, zs, target):
    deltas = [None] * len(weights)
    # Output layer error
    error = target - activations[-1]
    delta = error * sigmoid_derivative(activations[-1])
    deltas[-1] = delta

    #Backpropagate error to hidden layers
    for l in range(len(deltas)-2, -1, -1):
        delta = np.dot(deltas[l+1], weights[l+1].T) * sigmoid_derivative(activations[l+1])
        deltas[l] = delta

    return deltas

#Update weights and biases using deltas
def update_parameters(activations, deltas, learning_rate):
    for i in range(len(weights)):
        weights[i] += learning_rate * np.outer(activations[i], deltas[i])
        biases[i] += learning_rate * deltas[i]

#Single training iteration
activations, zs = forward_pass(inputs)
deltas = backward_pass(activations, zs, target)
update_parameters(activations, deltas, learning_rate)

print(f"\nOutput after forward pass: {activations[-1]}")
print(f"Error: {target - activations[-1]}")

print("\nUpdated weights and biases:")
for idx, (w, b) in enumerate(zip(weights, biases), start=1):
    print(f"Layer {idx} weights:\n{w}")
    print(f"Layer {idx} biases:\n{b}")
