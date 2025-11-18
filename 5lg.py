import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def train_perceptron(inputs, outputs, learning_rate=0.1, max_epochs=1000, error_threshold=0.01):
    input_nodes = inputs.shape[1]
    weights = np.random.uniform(-1, 1, input_nodes)
    bias = np.random.uniform(-1, 1)
    epoch = 0
    
    while epoch < max_epochs:
        total_error = 0
        for i in range(len(inputs)):
            linear_output = np.dot(inputs[i], weights) + bias
            output = sigmoid(linear_output)
            error = outputs[i] - output
            total_error += np.abs(error)
            d_output = error * sigmoid_derivative(output)
            weights += learning_rate * d_output * inputs[i]
            bias += learning_rate * d_output
        
        if total_error < error_threshold * len(inputs):
            break
        epoch += 1
    
    return weights, bias, epoch + 1

def predict(inputs, weights, bias):
    linear_output = np.dot(inputs, weights) + bias
    return sigmoid(linear_output)

and_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
and_outputs = np.array([0, 0, 0, 1])


or_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
or_outputs = np.array([0, 1, 1, 1])


print("\nTraining AND Gate.....")
and_weights, and_bias, and_epochs = train_perceptron(and_inputs, and_outputs)

print("\nTraining OR Gate.....")
or_weights, or_bias, or_epochs = train_perceptron(or_inputs, or_outputs)

print("\nAND Gate -")
print("Weights:", and_weights)
print("Bias:", and_bias)
print("Number of Epochs:", and_epochs)
print("Predictions:")
for i in range(len(and_inputs)):
    prediction = predict(and_inputs[i], and_weights, and_bias)
    print(f"Input: {and_inputs[i]} | Actual: {and_outputs[i]} | Predicted: {np.round(prediction)}")

print("\nOR Gate -")
print("Weights:", or_weights)
print("Bias:", or_bias)
print("Number of Epochs:", or_epochs)
print("Predictions:")
for i in range(len(or_inputs)):
    prediction = predict(or_inputs[i], or_weights, or_bias)
    print(f"Input: {or_inputs[i]} | Actual: {or_outputs[i]} | Predicted: {np.round(prediction)}")

