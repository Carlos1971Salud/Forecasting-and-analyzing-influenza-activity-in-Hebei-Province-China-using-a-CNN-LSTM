import pandas as pd
import matplotlib.pyplot as plt

# 假设数据保存在名为df的DataFrame中
df = pd.read_csv("C:/Users/administer/PycharmProjects/torch/result/2010-2022every_week.csv")

# 创建一个新的DataFrame，只包含train和test数据
train_data = df[df['set'] == 'train']
test_data = df[df['set'] == 'test']

# 绘制折线图
plt.figure(figsize=(16, 9))
plt.plot(train_data['weeks'], train_data['ILIp'], label='Train set (First 540 weeks)', color='darkblue')
plt.plot(test_data['weeks'], test_data['ILIp'], label='Test set (Last 136 weeks)', color='orange')

plt.axvline(x=540, color='red', linestyle='--', linewidth=0.5)

labels = [(540, 'Week 20th, 2021')]
for week, label_text in labels:
    plt.text(week, 5, label_text, fontsize=12, ha='center', va='bottom', color='black')

for i in range(len(df)):
    if (i + 1) % 52 == 0:  # 每一年有52个小格
        plt.axvline(x=i+1, color='#AFC7E8', linestyle='-', linewidth=0.5)  # 绘制竖直虚线，表示小刻度

labels = [(1, '2010'), (53, '2011'), (105, '2012'), (158, '2013'), (209, '2014'), (261, '2015'),
          (313, '2016'), (365, '2017'), (417, '2018'), (469, '2019'), (521, '2020'), (573, '2021'),
          (625, '2022')]
for week, label_text in labels:
    plt.text(week+10, 0.7, label_text, fontsize=9, ha='left', va='bottom', color='black')

# 添加标签和标题
plt.xlabel('Weeks')
plt.ylabel('ILI%')
# plt.title('ILI% Observation over Weeks')
plt.ylim(0.5, 6)
plt.legend()
plt.tight_layout()

# 显示图形
plt.show()
