import torch
import torch.nn as nn

# 입력 데이터 생성 (흑백 이미지 가정)
batch_size = 1
input_channels = 1
height = 32
width = 32


# 두 개의 컨볼루션 층을 가진 모델 생성
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# 모델 인스턴스 생성
model = ConvNet()

# 입력 데이터 생성
input_data = torch.randn(batch_size, input_channels, height, width)

# 각 층의 출력 확인
with torch.no_grad():
    print(f"입력 데이터 형태: {input_data.shape}")  # [1, 1, 32, 32]

    # 첫 번째 컨볼루션 층 통과
    conv1_output = model.conv1(input_data)
    print(f"\n첫 번째 컨볼루션 층 출력 형태: {conv1_output.shape}")  # [1, 64, 32, 32]

    # 첫 번째 층의 출력 채널들 확인
    print("\n첫 번째 층의 출력 (처음 3개 채널만):")
    for i in range(3):
        channel = conv1_output[0, i]  # 첫 번째 배치의 i번째 채널
        print(f"\n채널 {i + 1} 형태: {channel.shape}")  # [32, 32]
        print(f"채널 {i + 1} 첫 few 값:\n{channel[:3, :3]}")

    # 두 번째 컨볼루션 층 통과
    conv2_output = model.conv2(conv1_output)
    print(f"\n두 번째 컨볼루션 층 출력 형태: {conv2_output.shape}")  # [1, 128, 32, 32]

    # 두 번째 층의 출력 채널들 확인
    print("\n두 번째 층의 출력 (처음 3개 채널만):")
    for i in range(3):
        channel = conv2_output[0, i]  # 첫 번째 배치의 i번째 채널
        print(f"\n채널 {i + 1} 형태: {channel.shape}")  # [32, 32]
        print(f"채널 {i + 1} 첫 few 값:\n{channel[:3, :3]}")

# 각 컨볼루션 층의 필터 형태 확인
print(f"\n첫 번째 컨볼루션 층 필터 형태: {model.conv1.weight.shape}")  # [64, 1, 3, 3]
print(f"두 번째 컨볼루션 층 필터 형태: {model.conv2.weight.shape}")  # [128, 64, 3, 3]