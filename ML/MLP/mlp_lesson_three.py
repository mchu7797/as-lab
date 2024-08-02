import numpy as np
from mlp import MLP

X = np.array(
    [
        [1, 1, 1, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 1, 0, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 1, 0, 0, 1],
    ]
)
y = np.array(
    [
        [0.9, 0.1, 0.1],
        [0.1, 0.9, 0.1],
        [0.1, 0.1, 0.9],
    ]
)

mlp = MLP([9, 7, 5, 3])
mlp.train(X, y)

check_set = np.array(
    [
        [1, 1, 1, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 1, 0, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
    ]
)

predictions = mlp.forward(check_set)

np.set_printoptions(precision=4, suppress=True)

print("예측 결과 :")

for i in range(len(check_set)):
    print(
        f"입력: {check_set[i]}, 예측값: {predictions[i]}, 실제값: {y[i] if i < len(y) else None}"
    )
