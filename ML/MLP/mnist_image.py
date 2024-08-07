import os

import numpy as np
from PIL import Image


def digit_to_array(digit):
    array = np.array([0.1] * 10)
    array[digit % 10] = 0.9

    return array


def load_mnist(base_path, img_size=(28, 28)):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    # 훈련 데이터 로드
    train_path = os.path.join(base_path, 'trainingSet')
    for digit in range(10):
        digit_path = os.path.join(train_path, str(digit))
        for img_name in os.listdir(digit_path):
            img_path = os.path.join(digit_path, img_name)
            img = Image.open(img_path).convert('L')  # 그레이스케일로 변환
            img = img.resize(img_size)  # 이미지 크기 조정
            img_array = np.array(img) / 255.0  # 정규화
            train_data.append(img_array.flatten())  # 1차원 배열로 변환
            train_labels.append(digit_to_array(digit))

    # 테스트 데이터 로드
    test_path = os.path.join(base_path, 'testSet')
    for img_name in os.listdir(test_path):
        img_path = os.path.join(test_path, img_name)
        img = Image.open(img_path).convert('L')
        img = img.resize(img_size)
        img_array = np.array(img) / 255.0
        test_data.append(img_array.flatten())
        # 테스트 이미지의 레이블은 파일 이름에서 추출 (예: 'image_5.png'에서 5 추출)
        test_labels.append(int(img_name.split('_')[1].split('.')[0]))

    return (np.array(train_data), np.array(train_labels),
            np.array(test_data), np.array(test_labels))
