# 导入库
import torch
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 设置
warnings.filterwarnings('ignore')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 随机种子
torch.manual_seed(99)
np.random.seed(99)
random.seed(99)
print ("随机种子")

# 读取数据
train_df = pd.read_csv(r"C:\Users\administer\PycharmProjects\torch\HB_ILI_2010-2022.csv")
train_df.head()

ILI = train_df['spline_ILIp'].values
print(f"len(ILI):{len(ILI)}")
# plt.plot([i for i in range(len(ILI))], ILI)
# plt.show()

# 归一化
scaler = MinMaxScaler()
ILI = scaler.fit_transform(ILI.reshape(-1, 1))


# 数据集划分
def split_data(data, time_step=4):
    dataX = []
    dataY = []
    for i in range(len(data) - time_step):
        dataX.append(data[i:i + time_step])
        dataY.append(data[i + time_step])
    dataX = np.array(dataX).reshape(len(dataX), time_step, -1)
    dataY = np.array(dataY)
    return dataX, dataY


datax,datay = split_data(ILI, time_step=52)
print(f"dataX.shape:{datax.shape},datay.shape:{datay.shape}")


def train_test_split(dataX, datay, shuffle=True, percentage=0.8):
    if shuffle:
        random_num = [i for i in range(len(dataX))]
        np.random.shuffle(random_num)
        dataX = dataX[random_num]
        datay = datay[random_num]
    split_num = int(len(dataX) * percentage)
    train_X = dataX[:split_num]
    train_y = datay[:split_num]
    testX = dataX[split_num:]
    testy = datay[split_num:]
    return train_X, train_y, testX, testy


train_X, train_y, testX, testy = train_test_split(datax, datay, False, 0.8)
print(f"train_X.shape:{train_X.shape},test_X.shape:{testX.shape}")
X_train = train_X
y_train = train_y

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(y_train), torch.tensor(train_y))


# # 定义CNN模型
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv1d(52, 16, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
#         self.batchnorm = nn.BatchNorm1d(16)
#         self.dropout = nn.Dropout(p=0.2)
#         self.fc1 = nn.Linear(16 * 2, 1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.pool(x)
#         x = self.batchnorm(x)
#         x = self.dropout(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(52, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)  # 新添加的卷积层
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=1, padding=0)  # 新添加的池化层
        self.batchnorm = nn.BatchNorm1d(16)
        self.batchnorm2 = nn.BatchNorm1d(32)  # 针对第二个卷积层添加的批标准化层
        self.dropout = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(32, 1)  # 更新全连接层的输入维度

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.batchnorm(x)
        x = self.conv2(x)  # 新添加的卷积层
        x = self.relu(x)
        x = self.pool2(x)  # 新添加的池化层
        x = self.batchnorm2(x)  # 针对第二个卷积层添加的批标准化层
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


# 转换数据为Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(testX, dtype=torch.float32)
y_test_tensor = torch.tensor(testy, dtype=torch.float32)

# 初始化模型、损失函数和优化器
model = CNN()
# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 记录训练和测试损失
train_losses = []
test_losses = []

print("Start training")

# 训练模型
num_epochs = 500  # 训练周期数
batch_size = 32
# 批量大小

for epoch in range(num_epochs):
    # 在每个 epoch 开始时随机混淆训练集
    random_indices = torch.randperm(len(X_train))
    X_train_shuffled = X_train[random_indices]
    y_train_shuffled = y_train[random_indices]

    running_train_loss = 0.0

    # 批量训练
    for i in range(0, len(X_train), batch_size):
        # 获取当前批次的输入和标签
        inputs = X_train_shuffled[i:i + batch_size]
        labels = y_train_shuffled[i:i + batch_size]

        # 将输入和标签转换为张量
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        # 将梯度清空
        optimizer.zero_grad()
        # 计算模型的输出
        outputs = model(inputs_tensor)
        # 计算损失
        loss = criterion(outputs, labels_tensor)
        # 反向传播
        loss.backward()
        # 更新权重
        optimizer.step()

        # 累加训练损失
        running_train_loss += loss.item() * inputs_tensor.size(0)

    # 计算平均训练损失
    train_loss = running_train_loss / len(X_train)

    # 每20个 epoch 评估测试损失
    if epoch % 20 == 0:
        model.eval()  # 切换到评估模式
        with torch.no_grad():
            # 在测试集上评估模型-
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)
            train_losses.append(train_loss)
            test_losses.append(test_loss.item())

        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        model.train()  # 切换回训练模式

print("Size of test dataset:", len(X_test_tensor))
print("Length of train_losses:", len(train_losses))
print("Length of test_losses:", len(test_losses))

# 保存损失数据
loss_data = pd.DataFrame({
    'Epochs': range(0, num_epochs, 20),
    'Train Loss': train_losses,
    'Test Loss': test_losses
})
loss_data.to_csv('C:/Users/administer/PycharmProjects/torch/result/CNN_loss_data.csv', index=False)

print("Training complete!")

# 将损失值列表转换为NumPy数组
train_losses_np = np.array(train_losses)
test_losses_np = np.array(test_losses)

# 绘制损失函数图
plt.plot(train_losses_np, label='Training Loss')
plt.plot(test_losses_np, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss over Epochs')
plt.legend()
plt.show()


def mse(pred_y, true_y):
    return np.mean((pred_y-true_y) ** 2)


# 逆归一化&组合数据
train_pred = model(X_train_tensor).detach().numpy()
test_pred = model(X_test_tensor).detach().numpy()
pred_y = np.concatenate((train_pred, test_pred))
pred_y = scaler.inverse_transform(pred_y).T[0]
true_y = np.concatenate((y_train, testy))
true_y = scaler.inverse_transform(true_y).T[0]

# 结果写入
pred_result = pd.DataFrame({
    'pred_y': pred_y.flatten(),
    'true_y': true_y.flatten()
})
pred_result.to_csv('C:/Users/administer/PycharmProjects/torch/result/CNN_pred_result.csv', index=False)

plt.title("CNN for ILI% Prediction")
x = [i for i in range(len(true_y))]
plt.plot(x, pred_y, marker="D", markersize=1.5, color="blue", label="pred_y")
plt.plot(x, true_y, marker="o", markersize=1.5, color="purple", label="true_y")
plt.xlabel('Weeks')
plt.ylabel('ILI% Predicted Value')
plt.legend()
plt.show()

# MSE
print(f"mse_train(pred_y,true_y):{mse(train_pred, y_train)}")
print(f"mse_test(pred_y,true_y):{mse(test_pred, testy)}")
print(f"mse(pred_y,true_y):{mse(pred_y,true_y)}")

# RMSE
print(f"mean_squared_error_train:{mean_squared_error(train_pred, y_train, squared=False)}")
print(f"mean_squared_error_test:{mean_squared_error(test_pred, testy, squared=False)}")
print(f"mean_squared_error(pred_y, true_y):{mean_squared_error(pred_y,true_y, squared=False)}")

#MAE
print(f"mean_absolute_error_train:{mean_absolute_error(train_pred, y_train)}")
print(f"mean_absolute_error_test:{mean_absolute_error(test_pred, testy)}")
print(f"mean_absolute(pred_y, true_y):{mean_absolute_error(pred_y,true_y)}")

# R²
print(f"r2_score_train:{r2_score(train_pred, y_train)}")
print(f"r2_score_test:{r2_score(test_pred, testy)}")
print(f"r2_score(pred_y, true_y):{r2_score(pred_y,true_y)}")

# 将指标存储为字典
metrics_dict = {
    "Metric": ["MAE", "MSE", "RMSE", "R^2"],
    "Train": [mean_absolute_error(train_pred, y_train),
              mse(train_pred, y_train),
              mean_squared_error(train_pred, y_train, squared=False),
              r2_score(train_pred, y_train)],
    "Test": [mean_absolute_error(test_pred, testy),
             mse(test_pred, testy),
             mean_squared_error(test_pred, testy, squared=False),
             r2_score(test_pred, testy)]
}

# 写入CSV文件
with open('C:/Users/administer/PycharmProjects/torch/result/CNN_metrics.csv', 'w', newline='') as csvfile:
    fieldnames = ['Metric', 'Train', 'Test']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(len(metrics_dict["Metric"])):
        writer.writerow({'Metric': metrics_dict["Metric"][i],
                         'Train': metrics_dict["Train"][i],
                         'Test': metrics_dict["Test"][i]})
