import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 하이퍼파라미터 설정
input_size = 784  # 28x28
num_classes = 10
num_epochs = 3
batch_size = 100
learning_rate = 0.1

# MNIST 데이터셋 로드
# TODO : (0 ~ 1 사이의 값으로 수정)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# MLP 모델 정의
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()

        self.sigmoid = nn.Sigmoid()
        
        self.fc1 = nn.Linear(input_size, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

# 모델 인스턴스 생성
model = MLP(input_size, num_classes)

# 손실 함수와 옵티마이저 정의 (손실 함수: MSE Loss, 옵티마이저: 확률적 경사하강법)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 학습 루프
total_step = len(train_loader)
for epoch in range(num_epochs):
    total_loss = 0

    for i, (images, labels) in enumerate(train_loader):
        # 입력 데이터를 벡터로 변환
        images = images.reshape(-1, input_size)
        
        # 순전파
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}')

# 학습된 가중치와 편향을 파일로 저장
torch.save(model.state_dict(), 'mnist_mlp_model.db')
print("모델이 'mnist_mlp_model.db' 파일로 저장되었습니다.")

# 저장된 가중치와 편향 출력 (선택사항)
state_dict = model.state_dict()
for param_tensor in state_dict:
    print(param_tensor, "\t", state_dict[param_tensor].size())