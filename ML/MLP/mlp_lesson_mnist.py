import numpy as np

from mlp import MLP
from mnist_image import load_mnist

train_x, train_y, test_x, test_y = load_mnist("mnist")

mlp = MLP([784, 300, 100, 10])
mlp.train(train_x, train_y, epochs=10000, learning_rate=0.0001)

predictions = mlp.forward(test_x)

success = 0
fail = 0

for i in range(len(predictions)):
    # 예측된 클래스와 실제 클래스 비교
    predicted_class = np.argmax(predictions[i])
    true_class = np.argmax(test_y[i])
    if predicted_class == true_class:
        success += 1
    else:
        fail += 1

print(f"테스트 결과 : 실패 {fail}, 성공 {success}")
print(f"성공률 : {((success / (fail + success)) * 100):2f}")
