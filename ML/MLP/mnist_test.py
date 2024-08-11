import sys

import numpy as np
from PIL import Image

from mlp_cuda import MLP


def preprocess_image(image_path):
    # 이미지를 열고 그레이스케일로 변환
    img = Image.open(image_path).convert('L')
    # MNIST 데이터셋 크기(28x28)로 리사이즈
    img = img.resize((28, 28))
    # numpy 배열로 변환하고 정규화
    img_array = np.array(img) / 255.0
    # (1, 784) 형태로 펼치기
    return img_array.reshape(1, 784)


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 mnist_test.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    # MLP 모델 생성 (구조는 기존 코드와 일치해야 함)
    mlp = MLP([784, 300, 100, 10])

    try:
        # 저장된 가중치 로드
        mlp.load_weights("weights.numpy..bin")
    except FileNotFoundError:
        print("Error: weights.bin.bak file not found. Please train the model first.")
        sys.exit(1)

    # 이미지 전처리
    input_data = preprocess_image(image_path)

    # GPU로 데이터 이동
    input_data_gpu = np.array(input_data)

    # 예측
    output = mlp.forward(input_data_gpu)
    predicted_digit = np.argmax(output).get()

    print(f"The predicted digit is: {predicted_digit}")


if __name__ == "__main__":
    main()
