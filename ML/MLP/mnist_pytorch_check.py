import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 하이퍼파라미터 설정
input_size = 784  # 28x28
num_classes = 10
batch_size = 100

# MNIST 테스트 데이터셋 로드
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

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
        out = self.fc3(out)
        return out

# 모델 인스턴스 생성 및 저장된 가중치 로드
model = MLP(input_size, num_classes)
model.load_state_dict(torch.load('mnist_mlp_model.db', weights_only=True))
model.eval()  # 평가 모드로 설정

# 각 숫자별 정확도를 위한 변수 초기화
correct_pred = {classname: 0 for classname in range(10)}
total_pred = {classname: 0 for classname in range(10)}

# 테스트 데이터에 대한 정확도 평가
with torch.no_grad():
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        
        # 각 클래스별 정확도 계산
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[int(label)] += 1
            total_pred[int(label)] += 1

# 각 숫자별 정확도 출력
print("각 숫자별 정확도:")
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f"숫자 {classname} 정확도: {accuracy:.2f}%")

# 전체 정확도 계산 및 출력
total_correct = sum(correct_pred.values())
total_samples = sum(total_pred.values())
overall_accuracy = 100 * float(total_correct) / total_samples
print(f'전체 테스트 셋에 대한 정확도: {overall_accuracy:.2f}%')