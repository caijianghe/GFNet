import matplotlib.pyplot as plt

# 设置全局字体大小
plt.rcParams.update({'font.size': 12})

# 数据
x_labels = [1, 2, 3, 4, 5]
y_rmse = [5.15, 4.85, 4.78, 4.77, 4.77]
y_memory = [7.76, 8.80, 9.92, 11.66, 14.66]  # 示例数据，需要根据实际情况修改

# 自定义颜色
color_rmse = (1, 0.5, 0)  # RMSE 的颜色
color_memory = (0, 0.5, 0)  # Memory 的颜色

# 自定义标记样式
markers = ['o', '^', 's', 'D']

# 绘图
fig, ax1 = plt.subplots(figsize=(6, 5))  # 设置宽度为6英寸，高度为4英寸

# 绘制 RMSE 数据
ax1.plot(x_labels, y_rmse, label='RMSE', color=color_rmse, linestyle='-', marker=markers[0])
ax1.set_ylabel('RMSE (cm)')
ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

# 创建第二个 y 轴并绘制 Memory 数据
ax2 = ax1.twinx()
ax2.plot(x_labels, y_memory, label='Memory', color=color_memory, linestyle='-', marker=markers[1])
ax2.set_ylabel('Memory (G)')

# 设置 x 轴标签和标题


# 设置 x 轴刻度
plt.xticks(x_labels)
ax1.set_xlabel('stage (K)')
# 设置背景色为空白
plt.gca().set_facecolor('white')

# 添加图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

# 显示图形
plt.grid(True)
plt.savefig('curve_block.png', dpi=300)
plt.show()










# import matplotlib.pyplot as plt
#
# # 设置全局字体大小
# plt.rcParams.update({'font.size': 12})
#
# # 数据
# x_labels = [1, 2, 3, 4, 5]
# y_data = {
#     'NYU v2': [5.05, 4.83, 4.78, 4.77, 4.77],
#     'Middlebury': [3.10, 2.98, 2.95, 2.94, 2.94],
#     'RGB-D-D': [2.70, 2.58, 2.55, 2.54, 2.54],
#     # 'Implicit Neural': [4.82, 2.95, 2.55]
# }
#
# # 自定义颜色
# colors = [
#         (1, 0.5, 0),
#             (0.5, 0.5, 0.5),   # 浅橙色
#           (0.8, 0.8, 0), #
#           (0, 0.5, 0),   # 深绿色
#           ]  # 灰色
#
# # 自定义标记样式
# markers = ['o', '^', 's', 'D']
#
# # 绘图
# plt.figure(figsize=(6, 6))  # 设置宽度为6英寸，高度为6英寸
# for i, (label, data) in enumerate(y_data.items()):
#     plt.plot(x_labels, data, label=label, color=colors[i], linestyle= '-', marker=markers[i])
#
# # 设置图例、坐标轴和标题
# plt.legend(frameon=False, loc='upper right', bbox_to_anchor=(1, 1))
# plt.ylabel('RMSE (cm)')
# plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
# plt.xlabel('stage')
#
# plt.xticks(x_labels)
#
# # 移除网格线
# plt.grid(False)
#
# # 设置背景色为空白
# plt.gca().set_facecolor('white')
#
# # 显示图形
# plt.savefig('upsample.png', dpi=300)
# plt.show()

