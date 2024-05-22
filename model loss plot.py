import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 导入数据
df1 = pd.read_csv("C:/Users/administer/PycharmProjects/torch/result/CNN_loss.csv")
df2 = pd.read_csv("C:/Users/administer/PycharmProjects/torch/result/LSTM_loss.csv")
df3 = pd.read_csv("C:/Users/administer/PycharmProjects/torch/result/CNN-LSTM_loss.csv")

# 创建一个包含一行三列子图的布局
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# 绘制第一幅图
axs[0].plot(df1['Epochs'], df1['Train Loss'], label='Train Loss', marker='o')
axs[0].plot(df1['Epochs'], df1['Test Loss'], label='Test Loss', marker='o')
axs[0].set_title('CNN Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()
axs[0].set_ylim(0, 0.25)
axs[0].grid(False)

# 绘制第二幅图
axs[1].plot(df2['Epochs'], df2['Train Loss'], label='Train Loss', marker='o')
axs[1].plot(df2['Epochs'], df2['Test Loss'], label='Test Loss', marker='o')
axs[1].set_title('LSTM Loss')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Loss')
axs[1].legend()
axs[1].set_ylim(0, 0.25)
axs[1].grid(False)

# 绘制第三幅图
axs[2].plot(df3['Epochs'], df3['Train Loss'], label='Train Loss', marker='o')
axs[2].plot(df3['Epochs'], df3['Test Loss'], label='Test Loss', marker='o')
axs[2].set_title('CNN-LSTM Loss')
axs[2].set_xlabel('Epochs')
axs[2].set_ylabel('Loss')
axs[2].legend()
axs[2].set_ylim(0, 0.25)
axs[2].grid(False)

# 调整子图布局
plt.tight_layout()

# 显示图形
plt.show()
