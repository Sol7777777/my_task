# task1

## 初步搭建

### 数据加载和预处理

从OpenML数据库加载数据集，并分离特征X和目标变量y

然后利用StandardScaler方法对特征进行归一化，使其均值为 0，标准差为 1。这有利于增强模型的稳定性，加速收敛的速度

```py
from sklearn.datasets import fetch_openml
boston = fetch_openml(name="boston", version=1)
X = boston.data
y = boston.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 数据的划分

利用train_test_split的方法，将数据集以8:2的比例划分为训练集和测试集，使用随机数种子42

然后将训练集和测试集转化为tensor向量，方便进行后续的计算

```py
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
```

### 模型构建

nn.Module是 PyTorch 中所有神经网络模块的基类,它提供了一个方便的接口。我们可以利用它来构建自己的模型。

神经网络层，我采取三个全连接层，两个层之间用ReLU激活函数来激活：

输入层（13个特征）→ 隐藏层1（64个神经元）→ 隐藏层2（128个神经元）→ 输出层（1个神经元）

```py
class BostonHousingModel(nn.Module):
    def __init__(self):
        super(BostonHousingModel, self).__init__()
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        x = self.fc3(x)
        return x

model = BostonHousingModel()
```



### 损失函数和优化器选择

均方误差是回归问题中最常用的损失函数，适用于预测连续值，所以损失函数选择nn.MSELoss

在优化器的选择上，Adam有着自适应性和较快的收敛速度，所以选择Adam

### 训练和模型评估

训练函数的编写包含以下的步骤：

- 梯度清零：清零先前的梯度，防止干扰
- 前向传播：将输入传至模型得到预测输出
- 计算损失：计算预测输出与真实标签之间的损失值
- 反向传播：计算损失对模型参数的梯度
- 更新参数：更新模型的参数以优化模型

```py
def train():
    epoch_num = 500
    for epoch in range(epoch_num):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
```

在这之后，每50轮训练，打印出当前模型在训练集上的损失值，这是评判模型优劣的一个参数

```py
def test():
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        test_loss = criterion(predictions, y_test_tensor)
        print(f'Loss on testset: {test_loss.item():.4f}')
```

当模型经过训练便可以开始测试环节。

这里的model.eval()和训练函数中的model.train()告诉模型现在是训练模式还是评估模式，会影响到Dropout和BatchNorm的处理

最后打印出模型在测试集上的损失：

```py
Epoch [50/500], Loss: 255.1429
Epoch [100/500], Loss: 36.0972
Epoch [150/500], Loss: 21.2180
Epoch [200/500], Loss: 16.5008
Epoch [250/500], Loss: 13.6895
Epoch [300/500], Loss: 11.9649
Epoch [350/500], Loss: 10.8655
Epoch [400/500], Loss: 10.0425
Epoch [450/500], Loss: 9.3636
Epoch [500/500], Loss: 8.7579
Loss on testset: 12.2728
```



## 优化

目前的模型已经基本上实现了房价预测的任务，根据作答提示，对代码进行继续优化

### 数据清洗

- 处理缺失值：检查并处理数据中的缺失值。

  使用来自sklearn.impute的SimpleImputer方法，如果有存在缺失值，用当前特征的平均值来填充

  ```py
  imputer = SimpleImputer(strategy='mean')
  X = imputer.fit_transform(X)
  ```

- 处理异常值：极端的数据点，可能会对模型的训练产生负面影响。

  对异常值的处理，有许多的方法，比较简单易行的是直接删除，也可以将异常值替换成均值，中值等。按照直觉来想，这个数据集比较小，只有五百多条数据，如果直接删除可能会对训练造成过大的影响，所以我选择**四分位距法（IQR）**来处理异常值，将超过上限（Q3 + 1.5 * IQR）和低于下限（Q1 - 1.5 * IQR）的值视为异常值，并替换为上限和下限。（值得注意的是，13个特征中，"CHAS"是二值类型，不用进行数据清洗）

  ```py
  X_df = pd.DataFrame(X, columns=boston.feature_names)  
  
  for col in X_df.columns:
      if col == "CHAS":
          continue
      else:
          Q1 = X_cleaned_df[col].quantile(0.25)
          Q3 = X_cleaned_df[col].quantile(0.75)
          IQR = Q3 - Q1
          top = Q3 + 1.5 * IQR
          bot = Q1 - 1.5 * IQR
          value = X_cleaned_df[col].values
          value[value > top] = top
          value[value < bot] = bot
          X_cleaned_df[col] = value.astype(X_cleaned_df[col].dtype)
  ```
  
  最终在测试集上的损失：
  
  ```py
  Loss on testset: 13.3234
  ```
  
  然后又尝试了 **Z-score（标准化得分）**来检测异常值。保留所有特征的 Z-score 都小于阈值的样本，去除异常值。
  
  ```py
  z_scores = np.abs(stats.zscore(X_cleaned))
  threshold = 3
  X_no_outliers = X_cleaned[(z_scores < threshold).all(axis=1)]
  y_no_outliers = y[(z_scores < threshold).all(axis=1)]
  ```
  
  这里将阈值（threshold）设置为3，在特征中 **(值 - 均值) / 标准差** 与3的差值的绝对值大于这个阈值的，这个样本将被剔除。
  
  最终在测试集上的损失：
  
  ```py
  Loss on testset: 7.8953
  ```

### 数据集划分

在前完成的部分，将数据集以8：2划分为了训练集和测试集，根据做题提示，还需要得到验证集来进行超参数调优和模型的选择，

于是在先前的基础上，在训练集上继续以8：2划分为训练集和验证集

```py
# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_no_outliers, test_size=0.2, random_state=42)

# 将训练集划分为训练集和验证集
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
```



### 超参数调优和最佳模型保存

不同的超参数配置可以导致模型性能的显著差异。

我们希望在训练过程中，能通过验证验证集能找到最好的超参数组合。

以两个重要的超参数为例——学习率和隐藏层大小。

在原先的训练函数中，传入我们预定的超参数组合

其余的训练部分与第一次完成的保持不变，但在每50个epoch，评估一遍该超参数组合在验证集上的损失，并记录下来最小的损失，然后保存最小损失的模型

```py
def train(hidden_size1, hidden_size2, learning_rate):
    '''
    '''
    for epoch in range(epochs):
        '''
        '''
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
```

然后可以开始编写超参数调优函数

核心思想是遍历不同的超参数组合，得到每个组合经过500轮训练后能得到的最小损失，再比较每个组合的最小损失，最后得到最佳的超参数组合。

```py
def hyperparameter_tuning():
    hidden_sizes = [(64, 128), (128, 256)]
    learning_rates = [0.001, 0.01]
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
    return best_params
```

最后将最佳超参数组合传入test函数进行测试集上的计算，便可以得到最佳模型在测试集上的误差：

```py
Training with hidden sizes (64, 128) and learning rate 0.001
Epoch [50/500], Loss: 326.0832, Val Loss: 388.0172
'''
'''
Epoch [500/500], Loss: 7.7314, Val Loss: 15.8196
Training with hidden sizes (64, 128) and learning rate 0.01
Epoch [50/500], Loss: 12.7706, Val Loss: 22.9488
'''
'''
Epoch [500/500], Loss: 1.1501, Val Loss: 13.8579
Training with hidden sizes (128, 256) and learning rate 0.001
Epoch [50/500], Loss: 70.1041, Val Loss: 69.5037
'''
'''
Epoch [500/500], Loss: 5.4991, Val Loss: 13.5481
Training with hidden sizes (128, 256) and learning rate 0.01
Epoch [50/500], Loss: 11.0509, Val Loss: 20.1229
'''
'''
Epoch [500/500], Loss: 0.8396, Val Loss: 9.4200
Best parameters: ((128, 256), 0.01), Best validation loss: 9.4200
Loss on testset: 7.2935
```

### 房价区间分类及预测效果分析

先看了一眼数据集的房价大致分布，决定将最终的分类区间分为三类：低于15， 15到30， 高于30

然后确定最终目标，计算在落在各个区间的数据的MSE误差。

```py
price_ranges = {
    'Low price range': (y_true < 15),
    'Mid price range': ((y_true >= 15) & (y_true <= 30)),
    'High price range': (y_true > 30)
}
```

创建了一个字典，左边为价格描述，右边则是条件表达式，划分房价真实值的范围。

这里有一个布尔数组索引的概念，布尔数组是一个与原数组形状相同的数组，其中的每个元素是 `True` 或 `False`。使用这个布尔数组作为索引时，只有对应为 `True` 的位置的元素会被选中。

利用布尔数组索引，我们便可以将测试集以房价分划分标准，划分为三个区间，根据布尔数组索引位置的对应特性，便可以得到该区间上的均方误差。

```py
def analyze(y_true, y_pred):
    price_ranges = {
        '''
        '''
    }

    for range_name, condition in price_ranges.items():
        # 筛选出对应区间的真实值和预测值
        true_vals_in_range = y_true[condition]
        pred_vals_in_range = y_pred[condition]

        # 计算该区间的均方误差
        mse_in_range = criterion(true_vals_in_range, pred_vals_in_range)
        print(f'MSE in {range_name}: {mse_in_range:.4f}')
```

运行结果：

```bash

'''
'''
Training with hidden sizes (128, 256) and learning rate 0.01
Epoch [50/500], Loss: 11.4903, Val Loss: 20.0941
Epoch [100/500], Loss: 7.9462, Val Loss: 16.3280
Epoch [150/500], Loss: 6.3650, Val Loss: 14.9353
Epoch [200/500], Loss: 4.4099, Val Loss: 12.7586
Epoch [250/500], Loss: 2.6163, Val Loss: 11.0623
Epoch [300/500], Loss: 1.7441, Val Loss: 11.1356
Epoch [350/500], Loss: 1.2910, Val Loss: 12.2685
Epoch [400/500], Loss: 1.0142, Val Loss: 12.7979
Epoch [450/500], Loss: 0.7756, Val Loss: 13.3606
Epoch [500/500], Loss: 0.5939, Val Loss: 14.3775
Best parameters: ((128, 256), 0.01), Best validation loss: 11.0623
Loss on testset: 5.0868
MSE in Low price range: 3.2424
MSE in Mid price range: 4.8320
MSE in High price range: 9.1249
```

根据在三个区间上的误差，可以看出在房价的真实值小于30时，模型的预测效果较好，而房价真实值大于30时，模型的预测效果较前两者差一些。

## 改进方向

### 数据增强

从预测的效果来看，当房价的真实值较大时，模型的预测能力略为不足。猜测有某些特征在高房价区间的影响力较大，可以通过分析特征的重要性，对特征进行筛选，增强模型的预测能力。

### 超参数调优

模型只进行了学习率和隐藏层大小的组合，未来的改进，可以尝试加入更多超参数的组合，比如优化器的选择，激活函数，训练的轮次等

