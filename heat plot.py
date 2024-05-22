import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df1 = pd.read_csv("C:/Users/administer/PycharmProjects/torch/result/CNN_R2.csv")
df2 = pd.read_csv("C:/Users/administer/PycharmProjects/torch/result/LSTM_R2.csv")
df3 = pd.read_csv("C:/Users/administer/PycharmProjects/torch/result/CNN-LSTM_R2.csv")

# 创建子图，设置每个子图的大小和长宽比
fig, axes = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={'width_ratios': [1, 1, 1]})

# 第一张热图
heatmap_data1 = df1.pivot_table(index='batch_size', columns='learn_rate', values='R2')
sns.heatmap(heatmap_data1, annot=True, cmap='YlGnBu', fmt=".4f", ax=axes[0])
for i in range(len(heatmap_data1)):
    for j in range(len(heatmap_data1.columns)):
        axes[0].text(j + 0.5, i + 0.5, f"{heatmap_data1.iloc[i, j]:.4f}", ha='center', va='center', color='grey')
axes[0].set_title('CNN')
axes[0].set_xlabel('Learn Rate')
axes[0].set_ylabel('Batch Size')

# 第二张热图
heatmap_data2 = df2.pivot_table(index='batch_size', columns='learn_rate', values='R2')
sns.heatmap(heatmap_data2, annot=True, cmap='YlGnBu', fmt=".4f", ax=axes[1])
for i in range(len(heatmap_data2)):
    for j in range(len(heatmap_data2.columns)):
        axes[1].text(j + 0.5, i + 0.5, f"{heatmap_data2.iloc[i, j]:.4f}", ha='center', va='center', color='grey')
axes[1].set_title('LSTM')
axes[1].set_xlabel('Learn Rate')
axes[1].set_ylabel('Batch Size')

# 第三张热图
heatmap_data3 = df3.pivot_table(index='batch_size', columns='learn_rate', values='R2')
sns.heatmap(heatmap_data3, annot=True, cmap='YlGnBu', fmt=".4f", ax=axes[2])
for i in range(len(heatmap_data3)):
    for j in range(len(heatmap_data3.columns)):
        axes[2].text(j + 0.5, i + 0.5, f"{heatmap_data3.iloc[i, j]:.4f}", ha='center', va='center', color='grey')
axes[2].set_title('CNN-LSTM')
axes[2].set_xlabel('Learn Rate')
axes[2].set_ylabel('Batch Size')

plt.tight_layout()  # 调整子图布局，防止重叠
plt.show()
