"""
차량 이미지 분류를 위한 CNN 기반 딥러닝 모델 구현 모듈.

이 모듈은 차량 이미지를 분류하기 위한 딥러닝 시스템의 전체 파이프라인을 구현합니다:
- 데이터셋 관리 (VehicleDataset)
- CNN 모델 아키텍처 (VehicleClassifier)
- 모델 학습 및 검증 (VehicleClassifierTrainer)
- 모델 추론 (VehicleClassifierInference)

주요 기능:
- 이미지 데이터 로딩 및 전처리
- CNN 기반 특징 추출 및 분류
- 모델 학습, 검증, 체크포인트 저장
- 학습된 모델을 사용한 차량 유형 예측

사용 예시:
    # 학습 모드
    python car_cnn.py

    # 추론 모드
    python car_cnn.py --image path/to/image.jpg

Dependencies:
    - PyTorch
    - torchvision
    - Pillow
    - pandas
    - logging
"""

import os
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import resnet152, ResNet152_Weights
from PIL import Image
import pandas as pd
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# VehicleDataset: 차량 이미지와 라벨을 관리하는 커스텀 데이터셋 클래스
class VehicleDataset(Dataset):
    """
    차량 이미지와 라벨을 관리하는 커스텀 데이터셋.

    Attributes:
        data_dir (str): 데이터 디렉토리 경로
        data (pd.DataFrame): 이미지 파일명과 라벨이 포함된 데이터프레임
        transform (callable, optional): 이미지 전처리 및 증강을 위한 변환 함수
        class_to_idx (dict): 차량 유형과 인덱스 매핑 딕셔너리
    """
    def __init__(self, data_dir: str, csv_file: str, transform=None):
        """
        Args:
            data_dir: 데이터 디렉토리 경로
            csv_file: 이미지 파일명과 라벨이 포함된 CSV 파일
            transform: 이미지 전처리 및 증강을 위한 변환 함수
        """
        self.data_dir = data_dir
        self.data = pd.read_csv(os.path.join(data_dir, csv_file))
        self.transform = transform
        # 차량 유형을 숫자 인덱스로 매핑
        self.class_to_idx = {
            cls: idx for idx, cls in enumerate(self.data["vehicle_type"].unique())
        }

    def __len__(self):
        """데이터셋의 총 샘플 수를 반환합니다."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        지정된 인덱스의 이미지와 라벨을 반환합니다.

        Args:
            idx (int): 데이터 인덱스

        Returns:
            tuple: (이미지 텐서, 라벨 인덱스)

        Raises:
            Exception: 이미지 로딩 실패시 발생
        """
        try:
            img_path = os.path.join(
                self.data_dir, "images", self.data.iloc[idx]["image_filename"]
            )
            image = Image.open(img_path).convert("L")  # Grayscale로 변환
            label = self.class_to_idx[self.data.iloc[idx]["vehicle_type"]]

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            logger.error("Error loading image at index %s: %s", idx, e)
            raise

# VehicleClassifier: CNN 기반 차량 분류 모델
class VehicleClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super(VehicleClassifier, self).__init__()
        self.resnet = resnet152(weights=ResNet152_Weights.DEFAULT)

        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.resnet.conv1.out_channels,
            kernel_size=self.resnet.conv1.kernel_size,
            stride=self.resnet.conv1.stride,
            padding=self.resnet.conv1.padding,
            dilation=self.resnet.conv1.dilation,
            groups=self.resnet.conv1.groups,
            bias=False
        )
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# VehicleClassifierTrainer: 모델 학습을 관리하는 클래스
class VehicleClassifierTrainer:
    """
    모델 학습을 관리하는 클래스.

    Attributes:
        model (nn.Module): 학습할 모델
        device (torch.device): 학습에 사용할 디바이스
        config (dict): 학습 관련 설정값들
        criterion (nn.Module): 손실 함수
        optimizer (optim.Optimizer): 옵티마이저
        scheduler (optim.lr_scheduler._LRScheduler): 학습률 스케줄러
        best_val_acc (float): 최고 검증 정확도
        patience_counter (int): 조기 종료를 위한 카운터
    """
    def __init__(self, model, device, config: dict):
        """
        Args:
            model: 학습할 모델
            device: 학습에 사용할 디바이스 (CPU/GPU)
            config: 학습 관련 설정값들을 담은 딕셔너리
        """
        self.model = model
        self.device = device
        self.config = config
        self.best_val_acc = 0
        self.patience_counter = 0
        self._initialize_training()

    def _initialize_training(self):
        """학습에 필요한 손실 함수, 옵티마이저, 스케줄러 초기화"""
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"]
        )
        # 검증 손실이 개선되지 않으면 학습률 감소
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=3, factor=0.5
        )

    def train_epoch(self, train_loader):
        """한 에폭 동안의 학습을 수행"""
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0

        # tqdm으로 진행률 표시 추가
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # 현재 배치의 loss와 accuracy를 표시
            batch_acc = 100.0 * predicted.eq(labels).sum().item() / labels.size(0)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{batch_acc:.2f}%'
            })

        return running_loss / len(train_loader), 100.0 * correct / total

    def validate(self, val_loader):
        """검증 데이터로 모델 성능 평가"""
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0

        # 검증 과정도 진행률 표시
        pbar = tqdm(val_loader, desc='Validating', leave=False)
        with torch.no_grad():
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

                # 현재 배치의 loss와 accuracy를 표시
                batch_acc = 100.0 * predicted.eq(labels).sum().item() / labels.size(0)
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{batch_acc:.2f}%'
                })

        return running_loss / len(val_loader), 100.0 * correct / total

    def save_checkpoint(self, epoch, val_loss, val_acc):
        """모델 체크포인트 저장

        Args:
            epoch: 현재 에폭
            val_loss: 검증 손실
            val_acc: 검증 정확도
        """
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            self.config["model_save_path"],
        )
        logger.info("Model checkpoint saved with validation accuracy: %.2f%%", val_acc)

    def train(self, train_loader, val_loader, num_epochs):
        """전체 학습 과정 수행

        Args:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            num_epochs: 총 에폭 수
        """
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            logger.info(
                "Epoch %d: Train Loss: %.4f, Train Acc: %.2f%%, Val Loss: %.4f, Val Acc: %.2f%%",
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )

            # 학습률 조정 및 조기 종료 검사
            self.scheduler.step(val_loss)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, val_loss, val_acc)
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # 조기 종료 조건 확인
            if self.patience_counter >= self.config["patience"]:
                logger.info("Early stopping triggered")
                break


# 추론을 위한 클래스
class VehicleClassifierInference:
    """
    저장된 모델을 사용하여 추론을 수행하는 클래스.

    Attributes:
        device (torch.device): 추론에 사용할 디바이스
        class_mapping (dict): 클래스 인덱스-이름 매핑
        model (nn.Module): 로드된 모델
        transform (transforms.Compose): 이미지 전처리 변환
    """

    def __init__(self, model_path, device, class_mapping):
        """
        Args:
            model_path: 저장된 모델 경로
            device: 추론에 사용할 디바이스
            class_mapping: 클래스 인덱스-이름 매핑
        """
        self.device = device
        self.class_mapping = class_mapping
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()

    def _load_model(self, model_path):
        """
        저장된 모델을 로드합니다.

        Args:
            model_path (str): 모델 체크포인트 파일 경로

        Returns:
            nn.Module: 로드된 모델
        """
        model = VehicleClassifier(len(self.class_mapping))
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(self.device).eval()

    def _get_transform(self):
        """
        추론을 위한 이미지 전처리 변환을 반환합니다.

        Returns:
            transforms.Compose: 전처리 변환 파이프라인
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),  # Grayscale 변환 추가
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def predict(self, image_path):
        """이미지에 대한 차량 종류 예측

        Args:
            image_path: 예측할 이미지 경로

        Returns:
            str: 예측된 차량 종류
            float: 예측 신뢰도
        """
        image = Image.open(image_path).convert("L")  # Grayscale로 변환
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            confidence = probabilities[0][prediction].item()

        return self.class_mapping[prediction.item()], confidence


def main():
    """
    메인 실행 함수.

    커맨드 라인 인자를 파싱하고 모델의 학습 또는 추론을 실행합니다.
    --image 인자가 제공되면 추론 모드로, 그렇지 않으면 학습 모드로 동작합니다.
    """
    # 커맨드 라인 인자 파싱
    parser = argparse.ArgumentParser(description="Vehicle Classification")
    parser.add_argument("--image", type=str, help="Path to image for inference")
    args = parser.parse_args()

    # 설정값 정의
    config = {
        "data_dir": "car_train",
        "csv_file": "labels.csv",
        "model_save_path": "vehicle_classifier.pth",
        "batch_size": 32,
        "learning_rate": 0.001,
        "num_epochs": 30,
        "patience": 5,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.image:  # 추론 모드
        dataset = VehicleDataset(config["data_dir"], config["csv_file"])
        class_mapping = {v: k for k, v in dataset.class_to_idx.items()}
        inference = VehicleClassifierInference(
            config["model_save_path"], device, class_mapping
        )
        predicted_class, confidence = inference.predict(args.image)
        logger.info("Predicted: %s, Confidence: %.2f", predicted_class, confidence)
    else:  # 학습 모드
        # 학습용 데이터 변환
        transform_train = transforms.Compose([
            # 기본 크기 조정
            transforms.Resize((256, 256)),  # 약간 크게 리사이즈
            transforms.Grayscale(num_output_channels=1),  # Grayscale 변환 추가
            transforms.RandomCrop(224),  # 랜덤 크롭으로 최종 크기 조정

            # 기하학적 변환
            transforms.RandomHorizontalFlip(p=0.5),  # 좌우 반전
            transforms.RandomRotation(
                degrees=10,
                fill=255  # 흰색으로 채우기
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # 상하좌우 이동
                scale=(0.9, 1.1),  # 크기 변경
                shear=5,  # 기울이기
                fill=255  # 흰색으로 채우기
            ),

            # 밝기와 대비 조정 (흑백 이미지에 적합)
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),  # 선명도

            # 노이즈와 필터
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),  # 가우시안 블러

            # 기본 변환
            transforms.ToTensor(),

            # 흑백 이미지 정규화
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # 테스트용 데이터 변환
        transform_test = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Grayscale(num_output_channels=1),  # Grayscale 변환 추가
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        full_dataset = VehicleDataset(
            config["data_dir"], config["csv_file"], transform=transform_train
        )
        train_size = int(0.8 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        val_dataset.dataset.transform = transform_test
        test_dataset.dataset.transform = transform_test

        train_loader = DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config["batch_size"], shuffle=False
        )

        model = VehicleClassifier(len(full_dataset.class_to_idx)).to(device)
        trainer = VehicleClassifierTrainer(model, device, config)
        trainer.train(train_loader, val_loader, config["num_epochs"])


if __name__ == "__main__":
    main()