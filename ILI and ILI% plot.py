# 导入库
import pandas as pd
import matplotlib.pyplot as plt

# 假设你的 DataFrame 名称为 df
df = pd.read_csv("C:/Users/administer/PycharmProjects/torch/result/2010-2022ILIPcontinous.csv")

ILI = df['ILI']
ILIp = df['ILIp']

# 合并为一个新的 DataFrame
combined_df = pd.DataFrame({'ILI': ILI, 'ILIp': ILIp})

years = df['year'].unique()
weeks_per_year = len(df) // len(years)
weeks_labels = [f'{year}' for year in years]

# 创建图形和子图对象
fig, ax1 = plt.subplots(figsize=(16, 9))

# 绘制 ILI 曲线
ax1.plot(range(len(combined_df)), combined_df['ILI'], label='ILI', color='green')
ax1.set_xlabel('weeks')
ax1.set_ylabel('ILI', color='black')
ax1.set_xticks(range(0, len(combined_df), weeks_per_year))
ax1.set_xticklabels(weeks_labels, ha='center')
ax1.set_ylim(300, 5000)

# 创建第二个 y 轴
ax2 = ax1.twinx()
ax2.plot(range(len(combined_df)), combined_df['ILIp'], label='ILI percentage', color='orange')
ax2.set_ylabel('ILI percentage', color='black')
ax2.set_ylim(1.001, 5)

# 添加图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

for i in range(len(df)):
    if (i + 1) % 52 == 0:  # 每一年有52个小格
        plt.axvline(x=i, color='#AFC7E8', linestyle='--', linewidth=0.5)  # 绘制竖直虚线，表示小刻度

# plt.title('ILI and ILIp Over Observations')
plt.grid(False)
plt.tight_layout()
plt.show()
