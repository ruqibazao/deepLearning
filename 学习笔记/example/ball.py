# -*- coding: utf-8 -*-
# 特性归一化

import numpy as np

# 1. 读取数据
data = np.array(
    [
        [23, 12, 7, "胜"],
        [22, 10, 9, "败"],
        [26, 9, 12, "败"],
        [30, 9, 9, "胜"],
        [25, 3, 9, "败"],
        [27, 6, 9, "胜"],
        [21, 8, 8, "胜"],
        [23, 3, 7, "败"],
        [22, 4, 11, "胜"],
        [22, 6, 5, "胜"],
    ]
)

# 2. 将胜败转换为0和1
data[data=='胜'] = 1
data[data=='败'] = 0
# print(data)
data = data.astype(int)
print(data)

# 归一化
data_min = np.min(data, axis=0)
data_max = np.max(data, axis=0)
print('最小值：', data_min)
print('最大值：', data_max)
for i in range(3):
    data[:,i] = (data[:,i] - data_min[i]) / (data_max[i] - data_min[i])

print(data)