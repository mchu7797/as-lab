import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class SigmoidNeuron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = [random.uniform(-1, 1) for _ in range(input_size)]
        self.bias = random.uniform(-1, 1)
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def activate(self, inputs):
        weighted_sum = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return self.sigmoid(weighted_sum)

    def train(self, X, y, epochs=1000):
        for _ in range(epochs):
            for inputs, target in zip(X, y):
                prediction = self.activate(inputs)
                error = target - prediction
                delta = error * prediction * (1 - prediction)
                self.weights = [w + self.learning_rate * delta * x for w, x in zip(self.weights, inputs)]
                self.bias += self.learning_rate * delta


def train_gate(gate_type):
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    if gate_type == "AND":
        y = [0.1, 0.1, 0.1, 0.9]
    elif gate_type == "OR":
        y = [0.1, 0.9, 0.9, 0.9]
    elif gate_type == "XOR":
        y = [0.1, 0.9, 0.9, 0.1]
    elif gate_type == "XNAND":
        y = [0.9, 0.1, 0.1, 0.9]
    else:
        raise ValueError("Invalid gate type. Choose 'AND', 'OR', 'XOR', or 'XNAND'.")

    neuron = SigmoidNeuron(input_size=2)
    neuron.train(X, y)
    return neuron


def generate_test_cases():
    values = [round(i * 0.1, 1) for i in range(1, 10)]
    return [[x1, x2] for x1 in values for x2 in values]


def visualize_gate(gate_type, neuron):
    test_cases = generate_test_cases()
    X = [case[0] for case in test_cases]
    Y = [case[1] for case in test_cases]
    Z = [neuron.activate(case) for case in test_cases]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(X, Y, Z, c=Z, cmap='viridis')

    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')
    ax.set_zlabel('Output')
    ax.set_title(f'{gate_type} Gate')

    plt.colorbar(scatter)
    plt.show()


# Train and visualize each gate
for gate_type in ["AND", "OR", "XOR", "XNAND"]:
    neuron = train_gate(gate_type)
    visualize_gate(gate_type, neuron)