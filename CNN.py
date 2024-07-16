import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import itertools
import random
from tqdm import tqdm
import warnings  # 避免一些可以忽略的报错
warnings.filterwarnings('ignore')  # filterwarnings()方法是用于设置警告过滤器的方法，它可以控制警告信息的输出方式和级别.

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 导入Excel文件
file_path = "C:/Users/administer/Desktop/revision/ILI%.xlsx"
train_data = pd.read_excel(file_path, sheet_name='Sheet1')
test_data = pd.read_excel(file_path, sheet_name='Sheet2')

# 数据标准化
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)


# 数据准备
def create_sequences(data, time_steps=52):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)


time_steps = 52
X_train, y_train = create_sequences(train_scaled, time_steps)
X_test, y_test = create_sequences(test_scaled, time_steps)

# 转换为PyTorch的张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 确保输入的形状是 [batch_size, num_channels, sequence_length]
X_train = X_train.view(X_train.size(0), 1, X_train.size(1))
X_test = X_test.view(X_test.size(0), 1, X_test.size(1))


class CNN(nn.Module):
    def __init__(self, num_conv_layers, kernel_size, num_pool_layers, pool_size, dropout_rate=0.2):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList()
        input_channels = 1
        for _ in range(num_conv_layers):
            self.layers.append(nn.Conv1d(input_channels, 64, kernel_size, padding=kernel_size // 2))
            self.layers.append(nn.ReLU())
            input_channels = 64

        self.pool_layers = nn.ModuleList()
        for _ in range(num_pool_layers):
            self.pool_layers.append(nn.MaxPool1d(pool_size))

        self.fc1 = nn.Linear(64 * (time_steps // (pool_size ** num_pool_layers)), 50)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        for pool in self.pool_layers:
            x = pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


param_grid = {
    'num_conv_layers': [1, 2],
    'kernel_size': [3, 5],
    'num_pool_layers': [1, 2],
    'pool_size': [2],
    'learning_rate': [0.001, 0.01],
    'batch_size': [16, 32, 64]
}

best_params = None
best_loss = float('inf')

kf = KFold(n_splits=5)

for params in itertools.product(*param_grid.values()):
    params_dict = dict(zip(param_grid.keys(), params))
    fold_losses = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        model = CNN(params_dict['num_conv_layers'], params_dict['kernel_size'],
                    params_dict['num_pool_layers'], params_dict['pool_size'])

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params_dict['learning_rate'])
        batch_size = params_dict['batch_size']

        for epoch in tqdm(range(10),
                          desc=f'Fold {fold + 1}, Params {params_dict}'):
            model.train()
            permutation = torch.randperm(X_train_fold.size()[0])
            running_loss = 0.0
            for i in range(0, X_train_fold.size()[0], batch_size):
                indices = permutation[i:i + batch_size]
                batch_x, batch_y = X_train_fold[indices], y_train_fold[indices]

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / (X_train_fold.size()[0] // batch_size)
            print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')

        model.eval()
        val_outputs = model(X_val_fold)
        val_loss = criterion(val_outputs, y_val_fold).item()
        fold_losses.append(val_loss)

    avg_loss = np.mean(fold_losses)
    print(f'Params: {params_dict}, Avg Loss: {avg_loss:.4f}')
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_params = params_dict

print(f'Best Parameters: {best_params}, Best Loss: {best_loss}')

best_model = CNN(best_params['num_conv_layers'], best_params['kernel_size'],
                 best_params['num_pool_layers'], best_params['pool_size'])

criterion = nn.MSELoss()
optimizer = optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
batch_size = best_params['batch_size']
num_epochs = 200
dropout_rate = 0.2


for epoch in tqdm(range(num_epochs), desc='Training Best Model'):
    best_model.train()
    permutation = torch.randperm(X_train.size()[0])
    running_loss = 0.0
    for i in range(0, X_train.size()[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]

        optimizer.zero_grad()
        outputs = best_model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / (X_train.size()[0] // batch_size)
    print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

torch.save(best_model.state_dict(), 'best_cnn_model.pth')

best_model.eval()
y_pred = best_model(X_test).detach().numpy()

# 反归一化
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

# 计算MAE和MSE
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
print(f'MAE: {mae}, MSE: {mse}')

# 保存结果到Excel文件
result_df = pd.DataFrame({
    'True Value': y_test_rescaled.flatten(),
    'Predicted Value': y_pred_rescaled.flatten()
})
result_df.to_excel('CNN_result.xlsx', index=False)


# 绘制模型预测值和真实值的对比折线图
plt.figure(figsize=(14, 7))
plt.plot(y_test_rescaled, label='True Value')
plt.plot(y_pred_rescaled, label='Predicted Value')
plt.title('Comparison of True and Predicted Values')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
