import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt

class Config:
    batch_size = 64
    learning_rate = 0.01
    epochs = 25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.sigmoid1 = nn.Sigmoid()
        self.fc1 = nn.Linear(8 * 7 * 7, 128)
        self.sigmoid2 = nn.Sigmoid()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, save_path=None):
        x = self.conv1(x)
        if save_path:
            self.save_intermediate(x, save_path, "conv1")
        x = self.pool1(x)
        if save_path:
            self.save_intermediate(x, save_path, "pool1")
        x = self.conv2(x)
        if save_path:
            self.save_intermediate(x, save_path, "conv2")
        x = self.pool2(x)
        if save_path:
            self.save_intermediate(x, save_path, "pool2")
        x = self.flatten(x)
        x = self.sigmoid1(x)
        x = self.fc1(x)
        x = self.sigmoid2(x)
        x = self.fc2(x)
        return x

    def save_intermediate(self, x, save_path, phase):
        img = x.mean(dim=1).squeeze() if x.size(1) > 1 else x.squeeze()
        img = img.detach().cpu().numpy()
        plt.imshow(img, cmap='viridis')
        plt.axis('off')
        plt.colorbar()
        plt.savefig(f"{save_path}_{phase}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

class MNISTDataLoader:
    @staticmethod
    def check_data_availability():
        data_path = './data'
        train_data_path = os.path.join(data_path, 'MNIST', 'raw', 'train-images-idx3-ubyte')
        return os.path.exists(train_data_path)

    @staticmethod
    def load_data(config):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        return train_loader, test_loader

class ModelTrainer:
    def __init__(self, model, config):
        self.model = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
        self.config = config
        self.optimizer = optim.SGD(self.model.parameters(), lr=config.learning_rate, momentum=0.9)
        self.criterion = nn.MSELoss()

    def train(self, train_loader):
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.config.device), target.to(self.config.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            target_one_hot = torch.zeros(target.size(0), 10).to(self.config.device)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1)
            loss = self.criterion(output, target_one_hot)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {self.epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.config.device), target.to(self.config.device)
                output = self.model(data)
                target_one_hot = torch.zeros(target.size(0), 10).to(self.config.device)
                target_one_hot.scatter_(1, target.unsqueeze(1), 1)
                test_loss += self.criterion(output, target_one_hot).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

    def train_model(self, train_loader, test_loader):
        for self.epoch in range(1, self.config.epochs + 1):
            self.train(train_loader)
            self.test(test_loader)
        print("모델 학습 완료")
        torch.save(self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(), "mnist_cnn_model.pth")
        print("모델을 'mnist_cnn_model.pth'로 저장했습니다.")
        self.save_filters()

    def save_filters(self):
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        self.save_conv_filters(model.conv1, 'conv1_filters.png')
        self.save_conv_filters(model.conv2, 'conv2_filters.png')

    def save_conv_filters(self, conv_layer, save_path):
        weights = conv_layer.weight.data.cpu().numpy()
        num_filters = weights.shape[0]
        grid_size = int(np.ceil(np.sqrt(num_filters)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.flatten()
        for i in range(num_filters):
            ax = axes[i]
            im = ax.imshow(weights[i, 0], cmap='viridis')
            ax.axis('off')
            ax.set_title(f'Filter {i + 1}')
            plt.colorbar(im, ax=ax)
        for i in range(num_filters, len(axes)):
            fig.delaxes(axes[i])
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

class Predictor:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def predict_image(self, image_path):
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0).to(self.config.device)
        self.model.eval()
        with torch.no_grad():
            save_path = os.path.splitext(image_path)[0]
            output = self.model(image_tensor, save_path)
            prediction = output.argmax(dim=1, keepdim=True).item()
        return prediction

def main():
    config = Config()
    if len(sys.argv) == 1:  # 학습 모드
        if not MNISTDataLoader.check_data_availability():
            print("학습 데이터를 찾을 수 없습니다. 데이터를 다운로드합니다...")
        train_loader, test_loader = MNISTDataLoader.load_data(config)
        model = MNISTNet().to(config.device)
        trainer = ModelTrainer(model, config)
        trainer.train_model(train_loader, test_loader)
    elif len(sys.argv) == 2:  # 예측 모드
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"오류: 파일 '{image_path}'를 찾을 수 없습니다.")
            sys.exit(1)
        if not os.path.exists("mnist_cnn_model.pth"):
            print("오류: 학습된 모델을 찾을 수 없습니다. 먼저 모델을 학습시켜주세요.")
            sys.exit(1)
        model = MNISTNet().to(config.device)
        model.load_state_dict(torch.load("mnist_cnn_model.pth"))
        predictor = Predictor(model, config)
        predicted_digit = predictor.predict_image(image_path)
        print(f"입력 이미지: {image_path}")
        print(f"예측된 숫자: {predicted_digit}")
    else:
        print("사용법:")
        print("학습 모드: poetry run python mnist_cnn.py")
        print("예측 모드: poetry run python mnist_cnn.py <이미지_파일_경로>")
        sys.exit(1)

if __name__ == "__main__":
    main()
