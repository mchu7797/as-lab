import numpy as np

from mlp_cuda import MLP
from mnist_image import load_check_mnist


def accuracy(predictions, labels):
    return np.mean(np.argmax(predictions, axis=1) == labels)


# 테스트 데이터 로드
test_x, test_y = load_check_mnist("mnist", is_numpy=True)
print("테스트 이미지 로드 완료!")
print(f"로드된 테스트 이미지 수: {len(test_x)}")

# MLP 모델 로드
mlp = MLP([784, 300, 100, 10])
mlp.load_weights("weights.numpy")

# 예측 수행
predictions = mlp.forward(test_x)

# 전체 정확도 계산
acc = accuracy(predictions, test_y)
print(f"전체 테스트 정확도: {acc * 100:.2f}%")

# 각 숫자별 정확도 계산
for digit in range(10):
    digit_mask = (test_y == digit)
    digit_predictions = predictions[digit_mask]
    digit_labels = test_y[digit_mask]
    digit_acc = accuracy(digit_predictions, digit_labels)
    print(f"숫자 {digit}의 정확도: {digit_acc * 100:.2f}%")

# 오분류된 이미지 수 계산
misclassified = np.sum(np.argmax(predictions, axis=1) != test_y)
print(f"오분류된 이미지 수: {misclassified}")
print(f"오분류율: {misclassified / len(test_y) * 100:.2f}%")
