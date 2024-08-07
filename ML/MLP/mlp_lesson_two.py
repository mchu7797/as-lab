import numpy as np

from mlp import MLP

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.9, 0.1]])

mlp = MLP([2, 3, 2])
mlp.train(X, y)

np.set_printoptions(precision=2, suppress=True)

predictions = mlp.forward(X)
print("예측 결과:")
for i in range(len(X)):
    print(f"입력: {X[i]}, 예측값: {predictions[i]}, 실제값: {y[i]}")
