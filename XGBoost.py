import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt

# 加载数据
file_path = "C:\\Users\\administer\\Desktop\\revision\\ILI%.xlsx"
train_data = pd.read_excel(file_path, sheet_name='Sheet1', usecols=[0]).squeeze()
test_data = pd.read_excel(file_path, sheet_name='Sheet2', usecols=[0]).squeeze()

# 构建特征和目标变量的函数
def create_features(data, lag):
    X = np.array([data.shift(i) for i in range(lag, 0, -1)]).T
    y = data.values[lag:]
    return X[~np.isnan(X).any(axis=1)], y

# 设置滞后窗口为52
lag = 52

# 创建训练集的特征和目标变量
X_train, y_train = create_features(train_data, lag)

# 创建测试集的特征和目标变量
X_test, y_test = create_features(test_data, lag)

# 输出特征和目标变量的形状
print("训练集 X 的形状:", X_train.shape)
print("训练集 y 的形状:", y_train.shape)
print("测试集 X 的形状:", X_test.shape)
print("测试集 y 的形状:", y_test.shape)

# 定义XGBoost模型
model = xgb.XGBRegressor(objective='reg:squarederror', seed=42)

# 设置网格搜索的参数范围
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.7, 0.8, 0.9]
}

# 使用k折交叉验证，这里k设为5
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 定义均方误差评估器
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

# 实例化网格搜索
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring=mse_scorer, verbose=1)

# 在训练数据上拟合网格搜索
grid_search.fit(X_train, y_train)

# 获取最佳参数组合
best_params = grid_search.best_params_
print("最佳参数:", best_params)

# 获取所有参数组合的结果
cv_results = grid_search.cv_results_

# 计算每个参数组合在不同折上的平均损失
mean_test_scores = cv_results['mean_test_score']
params = cv_results['params']

for mean_score, param in zip(mean_test_scores, params):
    print(f"参数组合: {param}, 平均损失: {-mean_score:.4f}")

# 使用最佳参数重新训练模型
best_model = xgb.XGBRegressor(**best_params)
best_model.fit(X_train, y_train)

# 保存模型
joblib.dump(best_model, 'xgboost_model.pkl')

# 在测试集上进行预测
y_pred = best_model.predict(X_test)

# 计算MAE和MSE
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'MAE: {mae}, MSE: {mse}')

results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results_df.to_excel('XGBoost.xlsx', index=False)

# 绘制预测值和真实值的对比图
plt.figure(figsize=(10, 5))
plt.plot(range(len(y_test)), y_test, label='Actual')
plt.plot(range(len(y_pred)), y_pred, label='Predicted', linestyle='-')
plt.title('Comparison of Actual and Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.show()
