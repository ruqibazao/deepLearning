# # 导入模块  bp:前向计算
# import numpy as np 

# # 模拟生成10个用户的特征数据
# x = np.random.rand(10,5)
# print(x)

# # 模拟生成权重数据
# w = np.random.randint(1,10,size=5)

# # 定义一个偏置项
# b = np.random.randint(1,10,size=1)
# print(w)

# # 计算预测结果
# y = np.dot(x,w) + b

# print(y)

import numpy as np

def relu(x):
    """ReLU激活函数，用于添加非线性，确保输出非负."""
    return np.maximum(0, x)

def initialize_parameters(input_features, output_features):
    """初始化模型参数，包括权重和偏置."""
    W = np.random.rand(input_features, output_features)  # 权重初始化
    b = np.random.rand(output_features)                   # 偏置初始化
    return W, b

def predict_fare(input_data, W, b):
    """根据输入数据、权重和偏置预测打车费用."""
    z = np.dot(input_data, W) + b  # 线性部分: Wx + b
    output = relu(z)               # 应用ReLU激活函数
    return output

# 设置模型参数
input_features = 5  # 假设有5个特征：行驶时间、一天中的时间、行驶距离、区域、天气状况
output_features = 1  # 输出一个值，即预测的打车费用

# 初始化模型参数
W, b = initialize_parameters(input_features, output_features)

# 生成示例数据
# 每行代表一个样本，列对应不同的特征
input_data = np.random.rand(10, input_features)  # 生成10个样本

# 预测打车费用
predicted_fares = predict_fare(input_data, W, b)

# 打印预测结果
print("预测的打车费用:\n", predicted_fares)