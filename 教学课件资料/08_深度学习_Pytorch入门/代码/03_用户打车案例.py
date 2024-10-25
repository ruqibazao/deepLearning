import numpy as np 
import torch
from torch import nn

def load_data():
    # 构建简单的线性关系数据集
    data = np.array([
        [1.0, 15.0],
        [2.0, 18.0],
        [3.0, 21.0],
        [4.0, 24.0],
        [5.0, 27.0],
        [6.0, 30.0],
        [7.0, 33.0],
        [8.0, 36.0],
        [9.0, 39.0],
        [10.0, 42.0],
        [11.0, 45.0],
        [12.0, 48.0],
        [13.0, 51.0],
        [14.0, 54.0],
        [15.0, 57.0]
    ])
    # 指定训练集和测试集的比例
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    train_data = data[:offset]
    test_data = data[offset:]
    # 分割训练和测试数据集
    x_train_data = train_data[:, 0:1]
    y_train_data = train_data[:, 1:2]
    x_test_data = test_data[:, 0:1]
    y_test_data = test_data[:, 1:2]
    return x_train_data, y_train_data, x_test_data, y_test_data

# 调用load_data函数加载数据
x_train_data, y_train_data, x_test_data, y_test_data = load_data()


# 在Pytorch中, 需要处理的数据都是Tensor类型
x_train_tensor = torch.tensor(x_train_data,dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_data, dtype=torch.float32)

x_test_tensor = torch.tensor(x_test_data,dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_data, dtype=torch.float32)

# 定义网络模型 y = wx* b 线性关系
model = nn.Sequential(
    # 第一个参数: 代表输入的特征数量
    # 第二个参数: 需要有多少个输出
    # y1 = w1*x+b1
    # y2 = w2*x +b2
    nn.Linear(1,1)
)  # 序列网络模型结构

# 定义损失函数
loss_fn = nn.MSELoss()

# 定义一个优化器 随机梯度下降的优化器
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)


# 开始训练
for i in range(3000):
    #重置梯度数据
    model.zero_grad()
    # 前向计算 预测的过程
    y_pred = model(x_train_tensor)
    # 计算损失 Tensor
    loss = loss_fn(y_pred,y_train_tensor)
    # 计算梯度
    loss.backward()
    # 更新参数
    optimizer.step()
    print(f'训练轮次{i+1}, 损失率:{loss}')

# 打印模型参数
print(f'模型参数:{model.state_dict}')
print('='*100)
for name, param in model.named_parameters():
    print(f'{name}:{param}')


#模型评估
with torch.no_grad():
    y_pred = model(x_test_tensor)
    loss = loss_fn(y_pred,y_test_tensor)
    print(f'测试数据的损失:{loss}')