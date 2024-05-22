import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("C:/Users/administer/PycharmProjects/torch/result/ILI_result.csv")

# 创建一个空白的图表
plt.figure(figsize=(16, 9))

# 定义颜色和线条样式
colors = ['#E53528', 'darkblue', 'green', '#212F3D', 'orange']
linestyles = ['-', '-', '-', '-', '-']

# 绘制折线图
for i, model in enumerate(['SARIMA', 'CNN', 'LSTM', 'TRUE', 'CNN-LSTM']):
    model_data = df[(df['model'] == model)]
    if model in ['SARIMA', 'TRUE']:
        plt.plot(model_data['weeks'], model_data['pd'], label=model, color=colors[i],
                 linestyle=linestyles[i], linewidth=2)
    else:
        plt.plot(model_data['weeks'][52:], model_data['pd'][52:], label=model,
                 color=colors[i], linestyle=linestyles[i], linewidth=2)

for i in range(0, 157):
    if (i+1) % 52 == 0:  # 每一年有52个小格
        plt.axvline(x=i-19, color='#808B96', linestyle='--', linewidth=0.7)  # 绘制竖直虚线，表示小刻度
plt.axvline(x=1, color='#808B96', linestyle='--', linewidth=0.5)
# 标注位置和文本
labels = [(0, '2020,21th Week'), (32, '2021'), (84, '2022'), (136, '2023')]
for week, label_text in labels:
    plt.text(week, 0.7, label_text, fontsize=12, ha='left', va='bottom', color='black')

# 添加标签和标题
# plt.xlim(0, 136)
plt.xlabel('Weeks')
plt.ylabel('Predictions of ILI%')

plt.legend(loc='upper left', fontsize=13)

plt.ylim(0, 10)
plt.tight_layout()
# 显示图表
plt.grid(False)
plt.show()
