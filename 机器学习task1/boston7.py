import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from scipy import stats

# 加载数据集
boston = fetch_openml(name="boston", version=1)

X = boston.data
y = boston.target

# 数据清洗：处理缺失值
imputer = SimpleImputer(strategy='mean')
X_cleaned = imputer.fit_transform(X)

# 处理异常值（使用 Z-score）
z_scores = np.abs(stats.zscore(X_cleaned))
threshold = 3
X_no_outliers = X_cleaned[(z_scores < threshold).all(axis=1)]
y_no_outliers = y[(z_scores < threshold).all(axis=1)]

# 数据归一化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_no_outliers)

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_no_outliers, test_size=0.2, random_state=42)

# 将训练集划分为训练集和验证集
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train_final, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_final.values, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


# 定义神经网络模型
class BostonHousingModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(BostonHousingModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

criterion = nn.MSELoss()
# 训练模型，使用验证集进行评估
def train(hidden_size1, hidden_size2, learning_rate):
    model = BostonHousingModel(13, hidden_size1, hidden_size2)
      # 均方误差
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 500
    best_val_loss = float('inf')  # 用于记录最佳验证损失
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        # 每 50 个 epoch 评估一次验证集上的损失
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

            # 保存表现最好的模型
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                torch.save(model.state_dict(), 'best_model.pth')

    return best_val_loss


# 调整 hyperparameter_tuning 函数以返回最佳参数
def hyperparameter_tuning():
    hidden_sizes = [(64, 128), (128, 256)]  # 不同隐藏层大小
    learning_rates = [0.001, 0.01]  # 不同学习率
    best_loss = float('inf')
    best_params = None

    for hidden_size in hidden_sizes:
        for lr in learning_rates:
            print(f'Training with hidden sizes {hidden_size} and learning rate {lr}')
            val_loss = train(hidden_size[0], hidden_size[1], lr)
            if val_loss < best_loss:
                best_loss = val_loss
                best_params = (hidden_size, lr)

    print(f'Best parameters: {best_params}, Best validation loss: {best_loss:.4f}')
    return best_params  # 返回最佳参数

def analyze(y_true, y_pred):
    price_ranges = {
        'Low price range': (y_true < 15),
        'Mid price range': ((y_true >= 15) & (y_true <= 30)),
        'High price range': (y_true > 30)
    }

    for range_name, condition in price_ranges.items():
        # 筛选出对应区间的真实值和预测值
        true_vals_in_range = y_true[condition]
        pred_vals_in_range = y_pred[condition]

        # 计算该区间的均方误差
        mse_in_range = criterion(true_vals_in_range, pred_vals_in_range)
        print(f'MSE in {range_name}: {mse_in_range:.4f}')

# 测试模型
def test(best_params):  # 接收最佳参数作为参数
    model = BostonHousingModel(13, best_params[0][0], best_params[0][1])
    model.load_state_dict(torch.load('best_model.pth',weights_only=True))
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        test_loss = criterion(predictions, y_test_tensor)
        print(f'Test MSE: {test_loss.item():.4f}')
        # 分析不同房价区间的预测效果
        analyze(y_test_tensor, predictions)

if __name__ == "__main__":
    best_params = hyperparameter_tuning()  # 调优超参数并获取最佳参数
    test(best_params)  # 测试模型，传递最佳参数

