import time

from mlp_cuda import MLP
from mnist_image_cuda import load_mnist

train_x, train_y, test_x, test_y = load_mnist("mnist")
print("이미지 로드 완료!")

start_time = time.time()

mlp = MLP([784, 300, 100, 10])
mlp.train(train_x, train_y, epochs=50000, learning_rate=0.0001)

mlp.save_weights("weights.bin")

print("소요 시간(초) :", time.time() - start_time)