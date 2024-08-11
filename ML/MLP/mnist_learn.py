import time

from mlp_cuda import MLP
from mnist_image import load_mnist

train_x, train_y = load_mnist("mnist", is_numpy=False)
print("이미지 로드 완료!")

start_time = time.time()

mlp = MLP([784, 300, 100, 10])
mlp.train(train_x, train_y, epochs=30000, learning_rate=0.00008)

mlp.save_weights("weights.bin")
mlp.save_weights("weights.numpy")

print("소요 시간(초) :", time.time() - start_time)
