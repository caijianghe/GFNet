import matplotlib.pyplot as plt

# 设置全局字体大小
plt.rcParams.update({'font.size': 12})

# 数据
x_labels = ['NYUv2', 'Middlebury', 'RGB-D-D']
y_data = {
    r'GFNet w/ $L_{spa}$': [2.58, 1.70, 1.67],
    r'w/ $L_{gra}$': [2.54, 1.67, 1.66],
    r'w/ $L_{fre}$': [2.51, 1.65, 1.65],
    'All': [2.48, 1.64, 1.64]
}

# 自定义颜色
colors = [
            (0.5, 0.5, 0.5),   # 浅橙色
          (0.8, 0.8, 0), # 深黄色
          (0, 0.5, 0),   # 深绿色
          (1, 0.5, 0),]  # 灰色

# 自定义标记样式
markers = ['o', '^', 's', 'D']

# 绘图
plt.figure(figsize=(6, 6))  # 设置宽度为6英寸，高度为6英寸
for i, (label, data) in enumerate(y_data.items()):
    if label == r'w/ $L_{fre}$':
        linestyle = '-.'
    elif label == r'w/ $L_{gra}$':
        linestyle = '--'
    elif label == 'All':
        linestyle = '-'
    else:
        linestyle = ':'  # 指定 SDB1 为实线，其他为虚线
    plt.plot(x_labels, data, label=label, color=colors[i], linestyle=linestyle, marker=markers[i])

# 设置图例、坐标轴和标题
plt.legend(frameon=False)  # 移除图例框
plt.ylabel('RMSE (cm)')
plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
# plt.xlabel('Datasets')


# 移除网格线
plt.grid(False)

# 设置背景色为空白
plt.gca().set_facecolor('white')
plt.savefig('loss.png', dpi=300)
# 显示图形
plt.show()
