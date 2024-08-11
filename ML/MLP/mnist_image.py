import os

import cupy as cp
import numpy as np
from PIL import Image


def digit_to_array(digit):
    array = cp.array([0.1] * 10)
    array[digit % 10] = 0.9

    return array


def load_mnist(base_path, img_size=(28, 28), is_numpy=False):
    train_data = []
    train_labels = []

    # 훈련 데이터 로드
    for digit in range(10):
        digit_path = os.path.join(base_path, str(digit))
        for img_name in os.listdir(digit_path):
            img_path = os.path.join(digit_path, img_name)
            img = Image.open(img_path).convert('L')  # 그레이스케일로 변환
            img = img.resize(img_size)  # 이미지 크기 조정
            img_array = cp.array(img) / 255.0  # 정규화
            train_data.append(img_array.flatten())  # 1차원 배열로 변환
            train_labels.append(digit_to_array(digit))

    if is_numpy:
        return np.array(train_data), np.array(train_labels)
    else:
        return cp.array(train_data), cp.array(train_labels)


def load_check_mnist(base_path, img_size=(28, 28), samples_per_digit=1000, is_numpy=False):
    test_data = []
    test_labels = []

    for digit in range(10):
        digit_path = os.path.join(base_path, str(digit))
        img_files = os.listdir(digit_path)[:samples_per_digit]  # 각 숫자당 1000개씩 선택
        for img_name in img_files:
            img_path = os.path.join(digit_path, img_name)
            img = Image.open(img_path).convert('L')
            img = img.resize(img_size)
            img_array = cp.array(img) / 255.0
            test_data.append(img_array.flatten())
            test_labels.append(digit)

    if is_numpy:
        return np.array(test_data), np.array(test_labels)
    else:
        return cp.array(test_data), cp.array(test_labels)
