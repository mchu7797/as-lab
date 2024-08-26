import pickle
import cupy as cp

def sigmoid(x):
    return 1 / (1 + cp.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class MLP:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        # Xavier/Glorot 초기화 사용
        self.weights = []
        self.biases = []
        for i in range(self.num_layers - 1):
            limit = cp.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
            weight = cp.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1]))
            bias = cp.zeros((1, layer_sizes[i+1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, X):
        self.outputs = [X]
        for i in range(self.num_layers - 1):
            net_input = cp.dot(self.outputs[-1], self.weights[i]) + self.biases[i]
            output = sigmoid(net_input)
            self.outputs.append(output)
        return self.outputs[-1]

    def backward(self, X, y, learning_rate):
        self.outputs = [X]
        for i in range(self.num_layers - 1):
            net_input = cp.dot(self.outputs[-1], self.weights[i]) + self.biases[i]
            output = sigmoid(net_input)
            self.outputs.append(output)

        deltas = [None] * (self.num_layers - 1)
        deltas[-1] = (y - self.outputs[-1]) * sigmoid_derivative(self.outputs[-1])

        for i in range(self.num_layers - 2, 0, -1):
            deltas[i - 1] = deltas[i].dot(self.weights[i].T) * sigmoid_derivative(self.outputs[i])

        for i in range(self.num_layers - 1):
            self.weights[i] += learning_rate * self.outputs[i].T.dot(deltas[i])
            self.biases[i] += learning_rate * cp.sum(deltas[i], axis=0, keepdims=True)

    def train(self, X, y, epochs=10000, learning_rate=0.01, batch_size=32):
        n_samples = X.shape[0]
        for epoch in range(epochs):
            # 미니배치 학습
            for i in range(0, n_samples, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                self.backward(X_batch, y_batch, learning_rate)
            
            if (epoch + 1) % 5 == 0:
                loss = cp.mean(cp.square(y - self.forward(X)))
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:5f}")

    def save_weights(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.weights, self.biases), f)
        print(f"가중치와 편향이 {filename}에 저장되었습니다.")

    def load_weights(self, filename):
        with open(filename, 'rb') as f:
            loaded_weights, loaded_biases = pickle.load(f)
        if len(loaded_weights) != len(self.weights) or len(loaded_biases) != len(self.biases):
            raise ValueError("불러온 가중치와 편향의 구조가 현재 모델과 일치하지 않습니다.")
        self.weights = loaded_weights
        self.biases = loaded_biases
        print(f"가중치와 편향을 {filename}에서 불러왔습니다.")

    def export_to_numpy_weights(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((cp.asnumpy(self.weights), cp.asnumpy(self.biases)), f)
        print(f"NumPy 호환 가중치와 편향 데이터를 {filename}에 저장했습니다.")