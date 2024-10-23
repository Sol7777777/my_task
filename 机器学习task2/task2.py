import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, ReLU, Dropout, BatchNorm2d
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO:解释参数含义，在?处填入合适的参数
batch_size = 64 #每个批次的数据量设为64
learning_rate = 0.001 #学习率，就是模型学习的速度
num_epochs = 10 #训练的轮数

# 对输入的数据进行预处理，这里的ToTenser是把图片格式转化成tensor类型
transform = transforms.Compose([
    transforms.ToTensor()
])

# root可以换为你自己的路径
#训练数据集的下载和加载，加载时打乱顺序
trainset = torchvision.datasets.CIFAR10(root='dataset', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

#测试数据集的下载和加载，加载时不打乱顺序
testset = torchvision.datasets.CIFAR10(root='dataset', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # TODO:这里补全你的网络层
        self.model = Sequential(
            Conv2d(3, 32, 5, padding=2),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(2),

            Conv2d(32, 32, 5, padding=2),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(2),

            Conv2d(32, 64, 5, padding=2),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(2),

            Flatten(),
            Linear(64 * 4 * 4, 64),
            ReLU(),
            Dropout(0.5),
            Linear(64, 10)
        )

    def forward(self, x):
        # TODO:这里补全你的前向传播
        x = self.model(x)
        return x

# TODO:补全
#模型的实例化，并且使其转移到指定的设备上
model = Network().to(device)
#损失函数的选择，因为这里是十个标签的多分类问题，使用交叉熵损失函数，同样地，转移到指定的设备上
criterion = nn.CrossEntropyLoss().to(device)
#优化器的选择,10轮的小规模数据集的训练,Adam通常收敛更快，能迅速接近最优解
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#训练的函数
def train():
    model.train() #将模型转化成训练模式，会影响到Dropout和BatchNorm的处理
    for epoch in range(num_epochs): #在每一轮的训练开始前，把此轮的损失、分类正确的的样本数还有样本数初始化置零
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0): #开始遍历数据

            #获取到输入数据和对应的标签，并转移到指定的设备上
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            #清零之前的梯度，确保参数更新不受影响
            optimizer.zero_grad()

            #进行前向传播，获取对应的输出
            outputs = model(inputs)

            #通过损失函数，计算损失
            loss = criterion(outputs, labels)

            #反向传播，通过链式法则，计算每个模型参数的梯度
            loss.backward()

            #根据上一步得到的梯度，更新模型的参数
            optimizer.step()

            #损失的累加
            running_loss += loss.item()

            #torch.max返回两个参数，第一个是样本在每个类别上的最大值，第二个是最大值所在的类别的序号，我们只关心第二个，所以用"_"丢弃第一个返回值
            _, predicted = torch.max(outputs.data, 1)
            #样本数的累加
            total += labels.size(0)
            #分类正确的样本数的累加
            correct += (predicted == labels).sum().item()

        #计算这一轮训练的正确率并且打印
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%')

def test():
    model.eval() #将模型转化为评估模式，同样地，会影响Dropout和BatchNorm的处理
    #正确分类的样本数和总样本数初始化置零
    correct = 0
    total = 0

    with torch.no_grad(): #在测试环节，我们不希望去改变模型的参数，所以不用计算梯度，这样可以提高计算效率
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