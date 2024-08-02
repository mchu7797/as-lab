import numpy as np
from mlp import MLP

# 데이터 준비 (XOR 문제)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0.1], [0.9], [0.9], [0.1]])

# MLP 초기화 및 학습
mlp = MLP([2, 3, 1])
mlp.train(X, y)

np.set_printoptions(precision=2, suppress=True)

# 예측 및 결과 출력
predictions = mlp.forward(X)
print("예측 결과:")
for i in range(len(X)):
    print(f"입력: {X[i]}, 예측값: {predictions[i]}, 실제값: {y[i]}")
