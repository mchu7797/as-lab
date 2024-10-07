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

# 하이퍼파라미터 설정
batch_size = 64
learning_rate = 0.01
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_intermediate(x, save_path, phase):
    if phase in ['conv1', 'pool1']:
        num_filters = x.size(1)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for i in range(num_filters):
            img = x[0, i].detach().cpu().numpy()
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'Filter {i + 1}')

        plt.tight_layout()
        plt.savefig(f"{save_path}_{phase}.png")
        plt.close()

    elif phase in ['conv2', 'pool2']:
        num_filters = x.size(1)
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        axes = axes.flatten()

        for i in range(num_filters):
            img = x[0, i].detach().cpu().numpy()
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'Filter {i + 1}')

        plt.tight_layout()
        plt.savefig(f"{save_path}_{phase}.png")
        plt.close()


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.sigmoid1 = nn.Sigmoid()
        self.fc1 = nn.Linear(16 * 7 * 7, 128)
        self.sigmoid2 = nn.Sigmoid()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        intermediate = {}

        x = self.conv1(x)
        intermediate['conv1'] = x.clone()

        x = self.pool1(x)
        intermediate['pool1'] = x.clone()

        x = self.conv2(x)
        intermediate['conv2'] = x.clone()

        x = self.pool2(x)
        intermediate['pool2'] = x.clone()

        x = self.flatten(x)
        x = self.sigmoid1(x)
        x = self.fc1(x)
        x = self.sigmoid2(x)
        x = self.fc2(x)

        return x, intermediate


def check_data_availability():
    data_path = './data'
    train_data_path = os.path.join(data_path, 'MNIST', 'raw', 'train-images-idx3-ubyte')
    return os.path.exists(train_data_path)


def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data)

        target_one_hot = torch.zeros(target.size(0), 10).to(device)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)

        loss = criterion(output, target_one_hot)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)

            target_one_hot = torch.zeros(target.size(0), 10).to(device)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1)

            test_loss += criterion(output, target_one_hot).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(
        f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')


def predict_image(model, image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output, intermediate = model(image_tensor)
        prediction = output.argmax(dim=1, keepdim=True).item()

    save_path = os.path.splitext(image_path)[0]

    # 원본 이미지 저장
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title('Original Image')
    plt.savefig(f"{save_path}_original.png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # 중간 결과 저장
    for phase, tensor in intermediate.items():
        save_intermediate(tensor, save_path, phase)

    return prediction


def save_conv1_filters(model, save_path='conv1_filters.png'):
    conv1_weights = model.conv1.weight.data.cpu().numpy()

    num_filters = conv1_weights.shape[0]

    grid_size = int(np.ceil(np.sqrt(num_filters)))

    figure, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    axes = axes.flatten()

    for i in range(num_filters):
        ax = axes[i]
        im = ax.imshow(conv1_weights[i, 0], cmap='viridis')
        ax.axis('off')
        ax.set_title(f'Filter {i + 1}')
        plt.colorbar(im, ax=ax)

    for i in range(num_filters, len(axes)):
        figure.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)


def save_conv2_filters(model, save_path='conv2_filters.png'):
    conv2_weights = model.conv2.weight.data.cpu().numpy()

    num_filters = conv2_weights.shape[0]
    num_channels = conv2_weights.shape[1]

    fig, axes = plt.subplots(num_filters, num_channels, figsize=(num_channels * 3, num_filters * 3))

    for i in range(num_filters):
        for j in range(num_channels):
            im = axes[i, j].imshow(conv2_weights[i, j], cmap='viridis')
            axes[i, j].axis('off')
            axes[i, j].set_title(f'Filter {i + 1}, Channel {j + 1}')
            plt.colorbar(im, ax=axes[i, j])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_model(model, train_loader, test_loader):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader)

    print("모델 학습 완료")
    torch.save(model.state_dict(), "mnist_cnn_model.pth")
    print("모델을 'mnist_cnn_model.pth'로 저장했습니다.")

    save_conv1_filters(model)
    save_conv2_filters(model)


def main():
    if len(sys.argv) == 1:  # 인자가 없는 경우 (학습 모드)
        if not check_data_availability():
            print("학습 데이터를 찾을 수 없습니다. 데이터를 다운로드합니다...")

        train_loader, test_loader = load_data()
        model = MNISTNet().to(device)
        train_model(model, train_loader, test_loader)

    elif len(sys.argv) == 2:  # 이미지 파일 이름이 주어진 경우 (예측 모드)
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"오류: 파일 '{image_path}'를 찾을 수 없습니다.")
            sys.exit(1)

        if not os.path.exists("mnist_cnn_model.pth"):
            print("오류: 학습된 모델을 찾을 수 없습니다. 먼저 모델을 학습시켜주세요.")
            sys.exit(1)

        model = MNISTNet().to(device)
        model.load_state_dict(torch.load("mnist_cnn_model.pth", weights_only=True))
        predicted_digit = predict_image(model, image_path)
        print(f"입력 이미지: {image_path}")
        print(f"예측된 숫자: {predicted_digit}")
        print(f"중간 결과 이미지가 '{os.path.splitext(image_path)[0]}_*.png' 파일들로 저장되었습니다.")

    else:
        print("사용법:")
        print("학습 모드: poetry run python mnist_cnn.py")
        print("예측 모드: poetry run python mnist_cnn.py <이미지_파일_경로>")
        sys.exit(1)


if __name__ == "__main__":
    main()