import numpy as np


# 활성화 함수와 그 도함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# MLP 클래스
class MLP:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        # 가중치 초기화
        self.weights = []
        for i in range(self.num_layers - 1):
            weight = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1])
            self.weights.append(weight)

    def forward(self, X):
        self.outputs = [X]
        for i in range(self.num_layers - 1):
            net_input = np.dot(self.outputs[-1], self.weights[i])
            output = sigmoid(net_input)
            self.outputs.append(output)
        return self.outputs[-1]

    def backward(self, X, y, learning_rate):
        # 순전파 수행
        self.forward(X)

        # 역전파 계산
        deltas = [None] * (self.num_layers - 1)
        deltas[-1] = (y - self.outputs[-1]) * sigmoid_derivative(self.outputs[-1])

        for i in range(self.num_layers - 2, 0, -1):
            deltas[i - 1] = deltas[i].dot(self.weights[i].T) * sigmoid_derivative(
                self.outputs[i]
            )

        # 가중치 업데이트
        for i in range(self.num_layers - 1):
            self.weights[i] += self.outputs[i].T.dot(deltas[i]) * learning_rate

    def train(self, X, y, epochs=10000, learning_rate=0.1):
        for epoch in range(epochs):
            self.backward(X, y, learning_rate)
            if (epoch + 1) % 1000 == 0:
                loss = np.mean(np.square(y - self.outputs[-1]))
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
