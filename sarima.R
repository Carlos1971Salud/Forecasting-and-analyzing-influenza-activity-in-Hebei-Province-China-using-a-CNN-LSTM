# 导入工具包
library(ggplot2)
library(zoo)
library(forecast)
library(dplyr)
library(tseries)
library(aTSA)
library(lubridate)
library(caret)
library(caTools)
 
# 设置路径
getwd()
setwd("D:/Rfiles/study")

# 导入数据
data <- read.csv("ILI/HB_ILI_2010-2022N.csv",header = T)
head(data)
tail(data)

# 划分训练集和数据集
split <- 1:(nrow(data)*0.8)
trainset <- data[split,]
testset <- data[-split,]
dim(trainset)
dim(testset)
trainset <- trainset$NP
testset <- testset$NP

# 转化为时间序列类型
train_ts <- ts(trainset,frequency = 52,start = c(2010,1),end = c(2020,21))
test_ts <- ts(testset,frequency = 52,start = c(2020,21))
plot.ts(train_ts)
plot.ts(test_ts)

# 单位根检验
adf.test(train_ts,nlag = 3)
dftrain_ts <- diff(train_ts,differences = 52)
adf.test(dftrain_ts)

# 白噪声检验
for(k in 1:2) print(Box.test(train_ts,lag = 6*k,type = "Ljung-Box"))
for(k in 1:2) print(Box.test(dftrain_ts,lag = 6*k,type = "Ljung-Box"))

# 自相关图&偏自相关图
par(mfrow = c(2, 2))
acf(train_ts,lag.max = 104,main = "Raw Data ACF")
acf(dftrain_ts,lag.max = 104,main = "Differenced Data ACF")
pacf(train_ts,lag.max = 104,main = "Raw Data PACF")
pacf(dftrain_ts,lag.max = 104,main = "Differenced Data PACF")
par()

# 模型识别
auto.arima(train_ts)
fit <- arima(train_ts,order = c(2,0,0),seasonal = list(order=c(1,1,1),period = 52))
ts.diag(fit)
fit1 <- arima(train_ts,order = c(0,0,2),seasonal = list(order=c(1,1,0),period = 52))
ts.diag(fit1)
fit2 <- arima(train_ts,order = c(0,0,1),seasonal = list(order=c(1,1,0),period = 52))
ts.diag(fit2)
fit3 <- arima(train_ts,order = c(1,0,0),seasonal = list(order=c(1,1,0),period = 52))
ts.diag(fit3)
fit4 <- arima(train_ts,order = c(1,0,1),seasonal = list(order=c(1,1,0),period = 52))
ts.diag(fit4)
fit5 <- arima(train_ts,order = c(0,0,1),seasonal = list(order=c(1,1,0),period = 52))
ts.diag(fit5)
fit6 <- arima(train_ts,order = c(0,0,0),seasonal = list(order=c(0,1,0),period = 52))
ts.diag(fit6)
fit7 <- arima(train_ts,order = c(3,0,2),seasonal = list(order=c(0,1,0),period = 52))
ts.diag(fit7)
fit8 <- arima(train_ts,order = c(2,0,2),seasonal = list(order=c(1,1,1),period = 52))
ts.diag(fit8)
fit9 <- arima(train_ts,order = c(2,0,2),seasonal = list(order=c(1,1,0),period = 52))
ts.diag(fit9)
fit10 <- arima(train_ts,order = c(1,1,0),seasonal = list(order=c(0,0,2),period = 52))
ts.diag(fit10)
fit11 <- arima(train_ts,order = c(2,0,2),seasonal = list(order=c(0,1,0),period = 52))
ts.diag(fit11)

# 模型诊断
ts.diag(fit4)
ts.diag(fit10)

# 训练集预测
test_ts <- ts(testset,frequency = 52,start = c(2020,21))
pd_test <- forecast::forecast(train_ts,model=fit4,h=136)
pd_mean <- pd_test$mean
pd_test <- forecast::forecast(train_ts, model = fit4, h = 136)
pd_mean <- pd_test$mean
Lower_CI <- pd_test$lower[, "95%"]
Upper_CI <- pd_test$upper[, "95%"]

options(
  repr.plot.width = 16,
  repr.plot.height = 9
)

# 创建包含置信区间的数据框
comparison_data <- data.frame(
  Date = time(test_ts),  # 获取时间序列的日期
  Test = as.numeric(test_ts),  # 测试数据
  Prediction = pd_mean,  # 预测数据
  Lower_CI = Lower_CI,    # 置信区间下限
  Upper_CI = Upper_CI     # 置信区间上限
)

# 可视化
comparison_plot <- ggplot(comparison_data, aes(x = Date)) +
  geom_line(aes(y = Test, color = "True"), linewidth = 1.5) +  # 添加测试数据折线
  geom_line(aes(y = Prediction, color = "Forecasting Results"), linewidth = 1.5) +  # 添加预测数据折线
  geom_ribbon(aes(ymin = Lower_CI, ymax = Upper_CI), fill = "grey", alpha = 0.3) + 
  annotate("text", x = max(comparison_data$Date), y = max(comparison_data$Upper_CI), 
           label = "95% CI", hjust = 2.5, vjust = 10, col = "black", cex = 6) + # 添加置信区间
  labs(
    x = "week",  # x 轴标签
    y = "ILI%",  # y 轴标签
    color = ""  # 图例标题
  ) +
  scale_color_manual(values = c("True" = "blue", "Forecasting Results" = "red"))+# 设置线条颜色
  theme_bw() +  # 使用简约主题
  theme(legend.position = c(0.12, 0.88),legend.text = element_text(size = 11))
print(comparison_plot)

###############################


# 提取预测结果均值
# pd_mean <- test_forecast$mean

# 提取测试集真实值
true_values <- test_ts

# 计算误差
errors <- pd_mean - true_values

# 计算评价指标
MSE <- mean(errors^2)
MAE <- mean(abs(errors))
RMSE <- sqrt(MSE)

# 计算 R 方值
SS_res <- sum(errors^2)
SS_tot <- sum((true_values - mean(true_values))^2)
R_squared <- 1 - SS_res / SS_tot

# 创建包含结果的数据框
results <- data.frame(pd_mean = pd_mean)
metrics <- data.frame(MSE = MSE, MAE = MAE, RMSE = RMSE, R_squared = R_squared)

# 将结果写入 CSV 文件
write.csv(results, "D:/Rfiles/study/ILI/prediction_comparison.csv")
write.csv(metrics, "D:/Rfiles/study/ILI/evaluation_metrics.csv")


