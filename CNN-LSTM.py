import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings  # 避免一些可以忽略的报错
warnings.filterwarnings('ignore')  # filterwarnings()方法是用于设置警告过滤器的方法，它可以控制警告信息的输出方式和级别.

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 导入Excel文件
file_path = "C:\\Users\\administer\\Desktop\\revision\\ILI%.xlsx"
train_data = pd.read_excel(file_path, sheet_name='Sheet1')
test_data = pd.read_excel(file_path, sheet_name='Sheet2')

# 数据标准化
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)


# 创建LSTM输入和输出张量
def create_sequences(data, time_step=52):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 52
X_train, y_train = create_sequences(train_scaled, time_step)
X_test, y_test = create_sequences(test_scaled, time_step)

# 转换为PyTorch张量
X_train = torch.FloatTensor(X_train).unsqueeze(2)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
X_test = torch.FloatTensor(X_test).unsqueeze(2)
y_test = torch.FloatTensor(y_test).unsqueeze(1)

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# 定义CNN-LSTM模型
class CNNLSTM(nn.Module):
    def __init__(self, cnn_channels, lstm_hidden_size, num_layers, dropout_rate):
        super(CNNLSTM, self).__init__()
        self.cnn = nn.Conv1d(in_channels=1, out_channels=cnn_channels, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden_size, num_layers=num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x


# 训练和验证模型
def train_and_evaluate(train_loader, model, criterion, optimizer, num_epochs=200):
    model.train()
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    return model


# 超参数选择
dropout_rate = 0.2
param_grid = {
    'num_layers': [1, 2, 3],
    'hidden_size': [16, 32, 64],
    'batch_size': [32, 64, 128],
    'learn_rates': [0.001, 0.01]
}

best_loss = float('inf')

print(f"start")
kf = KFold(n_splits=5, shuffle=True)
results = []

for num_layers in param_grid['num_layers']:
    for hidden_size in param_grid['hidden_size']:
        for batch_size in param_grid['batch_size']:
            for learn_rates in param_grid['learn_rates']:
                fold_losses = []
                for fold, (train_index, val_index) in enumerate(kf.split(X_train)):

                    X_train_fold = X_train[train_index]
                    y_train_fold = y_train[train_index]

                    X_val_fold = X_train[val_index]
                    y_val_fold = y_train[val_index]

                    train_dataset_fold = TensorDataset(X_train_fold, y_train_fold)
                    train_loader_fold = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True)

                    val_dataset_fold = TensorDataset(X_val_fold, y_val_fold)
                    val_loader_fold = DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=True)


                    model = CNNLSTM(cnn_channels=16, lstm_hidden_size=hidden_size, num_layers=num_layers,
                                    dropout_rate=dropout_rate)
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rates)

                    model = train_and_evaluate(train_loader_fold, model, criterion, optimizer)

                    model.eval()
                    with torch.no_grad():
                        val_losses = []
                        for x_val, y_val in val_loader_fold:
                            val_outputs = model(x_val)
                            val_loss = criterion(val_outputs, y_val)
                            val_losses.append(val_loss.item())
                        average_val_loss = np.mean(val_losses)
                        fold_losses.append(average_val_loss)
                print(f'Fold {fold + 1}, Loss: {average_val_loss:.4f}')

                average_fold_performance = np.mean(fold_losses)
                results.append((num_layers, hidden_size, batch_size, learn_rates, average_fold_performance))
                print(
                    f'Params - Layers: {num_layers}, Hidden: {hidden_size}, Batch: {batch_size}, LR: {learn_rates}, Avg Loss: {average_fold_performance:.4f}')

# 找到最佳参数组合
best_params = min(results, key=lambda x: x[4])
print(
    f'Best Params - Layers: {best_params[0]}, Hidden: {best_params[1]}, Batch Size: {best_params[2]}, Learning Rate: {best_params[3]}, Loss: {best_params[4]:.4f}')

# 实例化模型
cnn_channels = 64
num_layers = best_params[0]
lstm_hidden_size = best_params[1]
batch_size = best_params[2]
lr = best_params[3]
dropout_rate = 0.2


model = CNNLSTM(cnn_channels=cnn_channels, lstm_hidden_size=lstm_hidden_size,
                num_layers=num_layers, dropout_rate=dropout_rate)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 将模型切换到训练模式
model.train()

# 设置训练参数
epochs = 200

# 训练模型
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, targets in train_loader:
        # 梯度置零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    # 打印每个epoch的训练损失
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')

torch.save(model.state_dict(), 'best_CNN-LSTM_model.pth')

print('Finished Training')

# 设置模型为评估模式
model.eval()

# 预测
with torch.no_grad():
    y_pred = model(X_test)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")


def inverse_transform(scaler, data):
    return scaler.inverse_transform(data)


# 计算指标
y_pred = y_pred.numpy()
y_test = y_test.numpy()

y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'MAE: {mae:.4f}, MSE: {mse:.4f}')

# 绘制折线图
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# 转换为 DataFrame
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
file_name = 'CNN-LSTM_result.xlsx'
df.to_excel(file_name, index=False)
print(f"Predictions and actuals saved to {file_name}")