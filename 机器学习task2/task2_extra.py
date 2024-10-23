import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, ReLU, Dropout, BatchNorm2d
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  # 导入可视化库

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64  # 每个批次的数据量设为64
learning_rate = 0.001  # 学习率，就是模型学习的速度
num_epochs = 50  # 训练的轮数

# 训练数据集的预处理
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616))
])

trainset = torchvision.datasets.CIFAR10(root='dataset', train=True, download=True, transform=train_transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

# 测试数据集的预处理（不进行数据增强）
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616))
])

testset = torchvision.datasets.CIFAR10(root='dataset', train=False, download=True, transform=test_transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = BatchNorm2d(out_channels)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = BatchNorm2d(out_channels)

        self.shortcut = Sequential()
        if in_channels != out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))) + self.shortcut(x)))

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = ResidualBlock(3, 64)
        self.layer2 = ResidualBlock(64, 128)
        self.layer3 = ResidualBlock(128, 256)
        self.layer4 = ResidualBlock(256, 512)
        self.pool = MaxPool2d(2)
        self.fc = Sequential(
            Flatten(),
            Linear(512 * 2 * 2, 512),
            ReLU(),
            Dropout(0.5),
            Linear(512, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.fc(x)
        return x


# 模型的实例化，并且使其转移到指定的设备上
model = Network().to(device)
# 损失函数的选择
criterion = nn.CrossEntropyLoss().to(device)
# 优化器的选择
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 用于存储损失和准确率
train_losses = []
train_accuracies = []

# 训练的函数
def train():
    model.train()  # 将模型转化成训练模式
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = running_loss / len(trainloader)

        # 记录损失和准确率
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

# 测试函数
def test():
    model.eval()  # 将模型转化为评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')

if __name__ == "__main__":
    train()
    test()

    # 绘制损失和准确率曲线
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy', color='orange')
    plt.title('Training Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()
