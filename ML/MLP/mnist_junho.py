import numpy as np
import cupy as cp
import os
from PIL import Image
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + cp.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def digit_to_array(digit):
    array = cp.array([0.1] * 10)
    array[digit % 10] = 0.9
    return array

def load_mnist(base_path, img_size=(28, 28)):
    train_data = []
    train_labels = []

    for digit in range(10):
        digit_path = os.path.join(base_path, str(digit))
        for img_name in os.listdir(digit_path):
            img_path = os.path.join(digit_path, img_name)
            img = Image.open(img_path).convert('L')
            img = img.resize(img_size)
            img_array = cp.array(img) / 255.0
            train_data.append(img_array.flatten())
            train_labels.append(digit_to_array(digit))

    return cp.array(train_data), cp.array(train_labels)

def load_check_mnist(base_path, img_size=(28, 28), samples_per_digit=1000):
    test_data = []
    test_labels = []

    for digit in range(10):
        digit_path = os.path.join(base_path, str(digit))
        img_files = os.listdir(digit_path)[:samples_per_digit]
        for img_name in img_files:
            img_path = os.path.join(digit_path, img_name)
            img = Image.open(img_path).convert('L')
            img = img.resize(img_size)
            img_array = cp.array(img) / 255.0
            test_data.append(img_array.flatten())
            test_labels.append(digit)

    return cp.array(test_data), cp.array(test_labels)

def BP2(inputs, targets, layer_num, Node, learning_rate, cycle, tolerance=1e-5, patience=5):
    weight = [cp.random.randn(Node[i], Node[i+1]) * 0.1 for i in range(layer_num-1)]
    bias = [cp.zeros((1, Node[i+1])) for i in range(layer_num-1)]

    best_weights = [w.copy() for w in weight]
    best_biases = [b.copy() for b in bias]
    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(cycle):
        total_loss = 0

        for i in range(len(inputs)):
            x = inputs[i]
            target = targets[i]

            # Forward propagation
            activations = [x]
            for k in range(layer_num-1):
                net_input = cp.dot(activations[k], weight[k]) + bias[k]
                activation = sigmoid(net_input)
                activations.append(activation)

            # Compute error
            output = activations[-1]
            error = target - output
            total_loss += cp.sum(cp.square(error))

            # Backpropagation
            delta = error * sigmoid_derivative(output)
            deltas = [delta]
            for k in range(layer_num-2, 0, -1):
                delta = cp.dot(deltas[-1], weight[k].T) * sigmoid_derivative(activations[k])
                deltas.append(delta)
            deltas.reverse()

            # Update weights and biases
            for k in range(layer_num-2, -1, -1):
                weight[k] += learning_rate * cp.dot(activations[k].reshape(-1, 1), deltas[k].reshape(1, -1))
                bias[k] += learning_rate * deltas[k]

        # Compute average loss
        avg_loss = total_loss / len(inputs)
        print(f"Epoch {epoch+1}/{cycle}, Loss: {avg_loss}")

        # Early stopping
        if avg_loss < best_loss - tolerance:
            best_loss = avg_loss
            best_weights = [w.copy() for w in weight]
            best_biases = [b.copy() for b in bias]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping")
                break

    print("Training complete")
    return best_weights, best_biases

def test(x, weights, biases):
    activations = [x]
    for k in range(len(weights)):
        net_input = cp.dot(activations[k], weights[k]) + biases[k]
        activation = sigmoid(net_input)
        activations.append(activation)
    return activations[-1]

# Set the base path for MNIST data
base_path = "mnist"

# Load and preprocess MNIST data
x_train, y_train = load_mnist(base_path)
x_test, y_test = load_check_mnist(base_path)

# Network configuration
totalnum = 4
num = [784, 300, 100, 10]
learning_rate = 0.1
cycle = 20  # Increased from 2 to 20 for better training

# Train the model
weights, biases = BP2(x_train, y_train, totalnum, num, learning_rate, cycle)

# Evaluate on training data
correct_predictions = sum(cp.argmax(test(x, weights, biases)) == cp.argmax(y) for x, y in zip(x_train, y_train))
accuracy = correct_predictions / len(x_train)
print(f"Accuracy on training data: {accuracy * 100:.2f}%")

# Evaluate on test data
correct_predictions = sum(cp.argmax(test(x, weights, biases)) == cp.argmax(y) for x, y in zip(x_test, y_test))
accuracy = correct_predictions / len(x_test)
print(f"Accuracy on test data: {accuracy * 100:.2f}%")