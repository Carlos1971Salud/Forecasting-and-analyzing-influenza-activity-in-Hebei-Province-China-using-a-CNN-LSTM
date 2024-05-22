# 导入库
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd  # 导入csv文件的库
import numpy as np  # 进行矩阵运算的库
import matplotlib.pyplot as plt  # 导入强大的绘图库
import torch   # 一个深度学习的库Pytorch
import torch.nn as nn  # neural network,神经网络
import torch.optim as optim  # 个实现了各种优化算法的库
from sklearn.preprocessing import MinMaxScaler
import random
import csv
import warnings  # 避免一些可以忽略的报错
warnings.filterwarnings('ignore')  # filterwarnings()方法是用于设置警告过滤器的方法，它可以控制警告信息的输出方式和级别.

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

train_df = pd.read_csv("C:/Users/administer/PycharmProjects/torch/HB_ILI_2010-2022N.csv")  # 导入文件
print(f"len(train_df):{len(train_df)}")  # 打印出train_df的长度
train_df.head()  # 展示前几行

ILI = train_df['spline_ILIp'].values
print(f"len(ILI):{len(ILI)}")
# plt.plot([i for i in range(len(ILI))], ILI)
# plt.show()

# 创建MinMaxScaler对象
scaler = MinMaxScaler()
ILI = scaler.fit_transform(ILI.reshape(-1, 1))


def split_data(data, time_step):
    dataX = []
    datay = []
    for i in range(len(data)-time_step):   # 取值从（数据长度-时间步长）
        dataX.append(data[i:i+time_step])  # 增加维度 i 到 i+t，为一个窗口，t=12，取0-11（12个）
        datay.append(data[i+time_step])  # 增加维度 i+t 为预测结果，取12（第13个）
    dataX = np.array(dataX).reshape(len(dataX), time_step, -1)
    datay = np.array(datay)
    return dataX, datay


dataX, datay = split_data(ILI, time_step=52)
print(f"dataX.shape:{dataX.shape},datay.shape:{datay.shape}")


# 划分训练集和测试集的函数
def train_test_split(dataX, datay, shuffle=True, percentage=0.8):
    """
    将训练数据X和标签y以numpy.array数组的形式传入
    划分的比例定为训练集:测试集=8:2
    """
    if shuffle:
        random_num = [index for index in range(len(dataX))]
        np.random.shuffle(random_num)
        dataX = dataX[random_num]
        datay = datay[random_num]
    split_num = int(len(dataX)*percentage)
    train_X = dataX[:split_num]
    train_y = datay[:split_num]
    test_X = dataX[split_num:]
    test_y = datay[split_num:]
    return train_X, train_y, test_X, test_y


train_X, train_y, test_X, test_y = train_test_split(dataX, datay, shuffle=False, percentage=0.8)
print(f"train_X.shape:{train_X.shape},test_X.shape:{test_X.shape}")
X_train, y_train = train_X, train_y  # 名称转换


# 定义CNN+LSTM模型类
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # 初始化隐藏状态h0
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # 初始化记忆状态c0
        # print(f"x.shape:{x.shape},h0.shape:{h0.shape},c0.shape:{c0.shape}")
        out, _ = self.lstm(x, (h0, c0))  # LSTM前向传播
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出作为预测结果
        return out


test_X1 = torch.Tensor(test_X)  # 转为torch的张量
test_y1 = torch.Tensor(test_y)

# 定义输入、隐藏状态和输出维度
input_size = 1  # 输入特征维度
hidden_size = 32  # LSTM隐藏状态维度
num_layers = 5  # LSTM层数
output_size = 1  # 输出维度（预测目标维度）

# 创建CNN_LSTM模型实例
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# 训练周期为500次
num_epochs = 500
batch_size = 256  # 一次训练的数量
# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.5, 0.999))
# 损失函数
criterion = nn.MSELoss()

train_losses = []
test_losses = []
print(f"start")

for epoch in range(num_epochs):

    random_num = [i for i in range(len(train_X))]
    np.random.shuffle(random_num)

    train_X = train_X[random_num]
    train_y = train_y[random_num]

    train_X1 = torch.Tensor(train_X[:batch_size])
    train_y1 = torch.Tensor(train_y[:batch_size])

    # 训练
    model.train()
    # 将梯度清空
    optimizer.zero_grad()
    # 将数据放进去训练
    output = model(train_X1)
    # 计算每次的损失函数
    train_loss = criterion(output, train_y1)
    # 反向传播
    train_loss.backward()
    # 优化器进行优化(梯度下降,降低误差)
    optimizer.step()

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            output = model(test_X1)
            test_loss = criterion(output, test_y1)
        train_losses.append(train_loss.detach())
        test_losses.append(test_loss.detach())
        print(f"epoch:{epoch},train_loss:{train_loss},test_loss:{test_loss}")


# 将训练集和测试集的损失数据存储到DataFrame中
loss_data = pd.DataFrame({
    'Epochs': range(0, num_epochs, 20),  # 每20个周期为一个间隔
    'Train Loss': train_losses,
    'Test Loss': test_losses
})

# 将DataFrame保存为CSV文件
loss_data.to_csv('C:/Users/administer/PycharmProjects/torch/result/LSTM_loss_data.csv', index=False)


# 均方误差计算
def mse(pred_y, true_y):
    return np.mean((pred_y-true_y) ** 2)


# 计算loss&绘制loss图
train_losses_np = [loss.item() for loss in train_losses]
test_losses_np = [loss.item() for loss in test_losses]
plt.figure(figsize=(10, 5))
plt.plot(range(0, num_epochs, 20), train_losses_np, label='Train Loss')
plt.plot(range(0, num_epochs, 20), test_losses_np, label='Test Loss')
plt.title('Training and Testing Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


train_X1 = torch.Tensor(X_train)
train_pred = model(train_X1).detach().numpy()
test_pred = model(test_X1).detach().numpy()
pred_y = np.concatenate((train_pred, test_pred))
pred_y = scaler.inverse_transform(pred_y).T[0]
true_y = np.concatenate((y_train, test_y))
true_y = scaler.inverse_transform(true_y).T[0]

# 结果写入
pred_result = pd.DataFrame({
    'pred_y': pred_y.flatten(),
    'true_y': true_y.flatten()
})
pred_result.to_csv('C:/Users/administer/PycharmProjects/torch/result/LSTM_pred_result.csv', index=False)

# MSE
print(f"mse_train(pred_y,true_y):{mse(train_pred, y_train)}")
print(f"mse_test(pred_y,true_y):{mse(test_pred, test_y)}")
print(f"mse(pred_y,true_y):{mse(pred_y,true_y)}")

# RMSE
print(f"mean_squared_error_train:{mean_squared_error(train_pred, y_train, squared=False)}")
print(f"mean_squared_error_test:{mean_squared_error(test_pred, test_y, squared=False)}")
print(f"mean_squared_error(pred_y, true_y):{mean_squared_error(pred_y,true_y, squared=False)}")

#MAE
print(f"mean_absolute_error_train:{mean_absolute_error(train_pred, y_train)}")
print(f"mean_absolute_error_test:{mean_absolute_error(test_pred, test_y)}")
print(f"mean_absolute(pred_y, true_y):{mean_absolute_error(pred_y,true_y)}")

# R²
print(f"r2_score_train:{r2_score(train_pred, y_train)}")
print(f"r2_score_test:{r2_score(test_pred, test_y)}")
print(f"r2_score(pred_y, true_y):{r2_score(pred_y,true_y)}")

# 将指标存储为字典
metrics_dict = {
    "Metric": ["MAE", "MSE", "RMSE", "R^2"],
    "Train": [mean_absolute_error(train_pred, y_train),
              mse(train_pred, y_train),
              mean_squared_error(train_pred, y_train, squared=False),
              r2_score(train_pred, y_train)],
    "Test": [mean_absolute_error(test_pred, test_y),
             mse(test_pred, test_y),
             mean_squared_error(test_pred, test_y, squared=False),
             r2_score(test_pred, test_y)],
    "Total": [mean_absolute_error(pred_y, true_y),
             mse(pred_y, true_y),
             mean_squared_error(pred_y, true_y, squared=False),
             r2_score(pred_y, true_y)]
    }

# 写入CSV文件
with open('C:/Users/administer/PycharmProjects/torch/result/LSTM_metrics.csv', 'w', newline='') as csvfile:
    fieldnames = ['Metric', 'Train', 'Test', 'Total']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(len(metrics_dict["Metric"])):
        writer.writerow({'Metric': metrics_dict["Metric"][i],
                         'Train': metrics_dict["Train"][i],
                         'Test': metrics_dict["Test"][i],
                         'Total': metrics_dict["Total"]})

# 绘图
plt.title("LSTM for ILI% Prediction")
x = [i for i in range(len(true_y))]
plt.plot(x, pred_y, marker="x", markersize=1.5, color="green", label="pred_y")
plt.plot(x, true_y, marker="o", markersize=1.5, color="purple", label="true_y")
plt.xlabel('Weeks')
plt.ylabel('ILI% Predicted Value')
plt.legend()
plt.show()
