import matplotlib.pyplot as plt
import matplotlib
# 设置全局字体大小
plt.rcParams.update({'font.size': 12})
# 汉字字体，优先使用楷体，找不到则使用黑体
plt.rcParams['font.sans-serif'] = ['SimHei']

# 正常显示负号
plt.rcParams['axes.unicode_minus'] = False


font_list = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
for i in font_list:
    print(i)
# 数据
x_labels = ['NYUv2', 'Middlebury', 'RGB-D-D']
y_data = {
    'Pixel-shuffle': [4.92, 2.99, 2.59],
    'Deconvolution': [4.96, 3.01, 2.59],
    'Up-projection': [4.86, 2.97, 2.56],
    '隐式神经插值': [4.82, 2.95, 2.55]
}

# 自定义颜色
colors = [
            (0.5, 0.5, 0.5),   # 浅橙色
          (0.8, 0.8, 0), #
          (0, 0.5, 0),   # 深绿色
          (1, 0.5, 0),]  # 灰色

# 自定义标记样式
markers = ['o', '^', 's', 'D']

# 绘图
plt.figure(figsize=(8, 6))  # 设置宽度为6英寸，高度为6英寸
for i, (label, data) in enumerate(y_data.items()):
    if label == '隐式神经插值':
        linestyle = '-'
    elif label == 'Deconvolution':
        linestyle = '--'
    elif label == 'Up-projection':
        linestyle = '-.'
    else:
        linestyle = ':'  # 指定 SDB1 为实线，其他为虚线
    plt.plot(x_labels, data, label=label, color=colors[i], linestyle=linestyle, marker=markers[i])

# 设置图例、坐标轴和标题
plt.legend(frameon=False)  # 移除图例框
plt.ylabel('均方根误差 RMSE (cm)')
plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
plt.xlabel('数据集')


# 移除网格线
plt.grid(False)

# 设置背景色为空白
plt.gca().set_facecolor('white')

# 显示图形
plt.savefig('upsample.png', dpi=600)
plt.show()
print("你好")

