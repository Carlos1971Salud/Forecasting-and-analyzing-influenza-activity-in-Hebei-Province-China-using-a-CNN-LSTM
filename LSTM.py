import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
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

# 加载数据
file_path = "C:\\Users\\administer\\Desktop\\revision\\ILI%.xlsx"
data_train = pd.read_excel(file_path, sheet_name='Sheet1', usecols=[0])
data_test = pd.read_excel(file_path, sheet_name='Sheet2', usecols=[0])

# 数据标准化
scaler = MinMaxScaler()
data_train_scaled = scaler.fit_transform(data_train)
data_test_scaled = scaler.transform(data_test)


# 创建数据集
def create_dataset(data, time_step=52):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)


X_train, y_train = create_dataset(data_train_scaled)
X_test, y_test = create_dataset(data_test_scaled)

# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# 初始化模型
model = LSTMModel(input_dim=1, hidden_dim=100, num_layers=2, dropout=0.2)

# 定义超参数
param_grid = {
    'num_layers': [1, 2, 3],
    'hidden_size': [16, 32, 64],
    'batch_size': [32, 64, 128],
    'learning_rate': [0.001, 0.01]
}

# 定义损失函数
criterion = nn.MSELoss()

# 准备5折交叉验证
kf = KFold(n_splits=5, shuffle=True)
results = []

for num_layers in param_grid['num_layers']:
    for hidden_size in param_grid['hidden_size']:
        for batch_size in param_grid['batch_size']:
            for learning_rate in param_grid['learning_rate']:
                fold_results = []
                for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
                    # 分割数据
                    X_train_fold = X_train[train_index]
                    y_train_fold = y_train[train_index]
                    X_val_fold = X_train[val_index]
                    y_val_fold = y_train[val_index]

                    # 数据加载器
                    train_dataset = TensorDataset(X_train_fold, y_train_fold)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                    val_dataset = TensorDataset(X_val_fold, y_val_fold)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                    # 初始化模型
                    model = LSTMModel(input_dim=1, hidden_dim=hidden_size, num_layers=num_layers, dropout=0.2)
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                    # 训练模型
                    for epoch in range(200):
                        model.train()
                        for x_batch, y_batch in train_loader:
                            optimizer.zero_grad()
                            output = model(x_batch)
                            loss = criterion(output, y_batch)
                            loss.backward()
                            optimizer.step()

                        # 验证损失
                        model.eval()
                        with torch.no_grad():
                            val_losses = []
                            for x_val, y_val in val_loader:
                                val_pred = model(x_val)
                                val_loss = criterion(val_pred, y_val)
                                val_losses.append(val_loss.item())
                            average_val_loss = np.mean(val_losses)
                            fold_results.append(average_val_loss)

                    print(f'Fold {fold + 1}, Loss: {average_val_loss:.4f}')

                # 计算并存储每种参数组合下的平均验证损失
                average_fold_performance = np.mean(fold_results)
                results.append((num_layers, hidden_size, batch_size, learning_rate, average_fold_performance))
                print(
                    f'Params - Layers: {num_layers}, Hidden: {hidden_size}, Batch: {batch_size}, LR: {learning_rate}, Avg Loss: {average_fold_performance:.4f}')

# 找到最佳参数组合
best_params = min(results, key=lambda x: x[4])
print(
    f'Best Params - Layers: {best_params[0]}, Hidden: {best_params[1]}, Batch Size: {best_params[2]}, Learning Rate: {best_params[3]}, Loss: {best_params[4]:.4f}')

input_dim = 1
num_layers = best_params[0]  # 最佳层数
hidden_dim = best_params[1]  # 最佳隐藏层大小
batch_size = best_params[2]  # 最佳批处理大小
learning_rate = best_params[3]  # 最佳学习率


# 初始化模型
model = LSTMModel(input_dim, hidden_dim, num_layers, dropout=0.2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 数据加载器
train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 训练模型
for epoch in range(200):
    model.train()
    running_loss = 0.0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x_batch.size(0)

    # 打印每个epoch的训练损失
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch + 1}/{epoch}], Loss: {epoch_loss:.4f}')

# 测试评估
test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for x_test, y_test in test_loader:
        pred = model(x_test)
        predictions.extend(pred.flatten().tolist())
        actuals.extend(y_test.flatten().tolist())

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

torch.save(model.state_dict(), 'best_LSTM_model.pth')

# 反归一化
predictions_scaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
actuals_scaled = scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

# 计算评估指标
mse = mean_squared_error(actuals_scaled, predictions_scaled)
mae = mean_absolute_error(actuals_scaled, predictions_scaled)
print(f'MSE: {mse}, MAE: {mae}')

# 绘制曲线
plt.figure(figsize=(10, 5))
plt.plot(actuals_scaled, label='Actual')
plt.plot(predictions_scaled, label='Predicted')
plt.title('Actual vs Predicted')
plt.xlabel('Samples')
plt.ylabel('Values')
plt.legend()
plt.show()

df = pd.DataFrame({'Predictions': predictions_scaled, 'Actuals': actuals_scaled})
file_name = 'LSTM_result.xlsx'
df.to_excel(file_name, index=False)
print(f"Predictions and actuals saved to {file_name}")
