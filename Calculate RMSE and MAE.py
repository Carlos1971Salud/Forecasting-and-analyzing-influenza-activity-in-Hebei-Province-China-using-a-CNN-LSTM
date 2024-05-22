import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 读取数据
data = pd.read_excel(r"C:\Users\administer\PycharmProjects\torch\result\conclusion for 4 models.xlsx")

# 提取真实值和预测值列
true_values = data['a']
predictions_b = data['b']
predictions_c = data['c'][52:]  # 从第53天开始
predictions_d = data['d'][52:]  # 从第53天开始
predictions_e = data['e'][52:]  # 从第53天开始

# 计算MSE、RMSE、MAE和R方
mse_b = mean_squared_error(true_values, predictions_b)
rmse_b = np.sqrt(mse_b)
mae_b = mean_absolute_error(true_values, predictions_b)
r2_b = r2_score(true_values, predictions_b)

mse_c = mean_squared_error(true_values[52:], predictions_c)
rmse_c = np.sqrt(mse_c)
mae_c = mean_absolute_error(true_values[52:], predictions_c)
r2_c = r2_score(true_values[52:], predictions_c)

mse_d = mean_squared_error(true_values[52:], predictions_d)
rmse_d = np.sqrt(mse_d)
mae_d = mean_absolute_error(true_values[52:], predictions_d)
r2_d = r2_score(true_values[52:], predictions_d)

mse_e = mean_squared_error(true_values[52:], predictions_e)
rmse_e = np.sqrt(mse_e)
mae_e = mean_absolute_error(true_values[52:], predictions_e)
r2_e = r2_score(true_values[52:], predictions_e)

# 打印结果
print("Metrics for predictions b:")
print("MSE:", mse_b)
print("RMSE:", rmse_b)
print("MAE:", mae_b)
print("R-squared:", r2_b)
print()

print("Metrics for predictions c:")
print("MSE:", mse_c)
print("RMSE:", rmse_c)
print("MAE:", mae_c)
print("R-squared:", r2_c)
print()

print("Metrics for predictions d:")
print("MSE:", mse_d)
print("RMSE:", rmse_d)
print("MAE:", mae_d)
print("R-squared:", r2_d)
print()

print("Metrics for predictions e:")
print("MSE:", mse_e)
print("RMSE:", rmse_e)
print("MAE:", mae_e)
print("R-squared:", r2_e)