# 导入库
import pandas as pd
import matplotlib.pyplot as plt

# 导入数据
df1 = pd.read_csv("C:/Users/administer/PycharmProjects/torch/result/2010-2022ILIPcontinous.csv")

# 创建绘图对象
plt.figure(figsize=(16, 9))

# 循环遍历每年的数据，并绘制折线图
for year, data in df1.groupby('year'):
    plt.plot(range(52), data['ILIp'], label=str(year))  # 绘制折线图，并以年份作为标签

# 添加图例
plt.legend(title='Year')

# 坐标轴
plt.xticks(range(1, 53, 1), rotation=90)
plt.set_ylim = (0, 10)

# 添加标题和轴标签
plt.xlabel('weeks')
plt.ylabel('ILI%')
plt.legend(title='Year', bbox_to_anchor=(0, 1.00), loc='upper left', ncol=3)

# 显示图形
plt.tight_layout()
plt.show()
#
# 导入数据
df2 = pd.read_csv("C:/Users/administer/PycharmProjects/torch/result/2010-2022ILIVcontinous.csv")

# 创建绘图对象
plt.figure(figsize=(16, 9))

# 第一幅图片
plt.subplot(1, 2, 1)  # 1行2列布局，第一张图
for year, data in df2.groupby('year'):
    plt.plot(range(52), data['ILI'], label=str(year))  # 绘制折线图，并以年份作为标签
plt.title('ILI')  # 添加标题
plt.xlabel('weeks')
plt.ylabel('ILI')
plt.ylim(300, 5499)
plt.legend(title='Year', bbox_to_anchor=(0, 1.01), loc='upper left', ncol=3)

# 第二幅图片
plt.subplot(1, 2, 2)  # 1行2列布局，第二张图
# 绘制第二幅图片的代码，替换下面的示例
for year, data in df1.groupby('year'):
    plt.plot(range(52), data['ILIp'], label=str(year))  # 绘制折线图，并以年份作为标签
plt.title('ILI%')  # 添加标题
plt.xlabel('weeks')
plt.ylabel('ILI%')
plt.ylim(1.000001, 5.499999)
plt.legend(title='Year', bbox_to_anchor=(0, 1.01), loc='upper left', ncol=3)

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()
