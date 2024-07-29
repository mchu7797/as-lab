import math
import random


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

                # Delta rule
                delta = error * prediction * (1 - prediction)

                # Update weights and bias
                self.weights = [w + self.learning_rate * delta * x for w, x in zip(self.weights, inputs)]
                self.bias += self.learning_rate * delta


def train_and_test_gate(gate_type):
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    if gate_type == "AND":
        y = [0.1, 0.1, 0.1, 0.9]
    elif gate_type == "OR":
        y = [0.1, 0.9, 0.9, 0.9]
    else:
        raise ValueError("Invalid gate type. Choose 'AND', 'OR', 'XOR', or 'XNAND'.")

    neuron = SigmoidNeuron(input_size=2)
    neuron.train(X, y)

    print(f"\n{gate_type} Gate Results:")
    correct_predictions = 0
    for inputs, target in zip(X, y):
        prediction = neuron.activate(inputs)
        predicted_class = 'TRUE' if prediction > 0.5 else 'FALSE'
        actual_class = 'TRUE' if target > 0.5 else 'FALSE'
        is_correct = predicted_class == actual_class
        correct_predictions += is_correct
        print(f"Input: {inputs}, Target: {target:.1f}, Prediction: {prediction:.4f}, "
              f"Predicted: {predicted_class}, Actual: {actual_class}, Correct: {is_correct}")

    accuracy = correct_predictions / len(X) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")

    print('\n', '-' * 50, sep='')

    return neuron


# Train and test AND gate
and_neuron = train_and_test_gate("AND")

# Train and test OR gate
or_neuron = train_and_test_gate("OR")