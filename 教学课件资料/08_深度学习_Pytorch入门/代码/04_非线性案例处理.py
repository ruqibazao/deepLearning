import numpy as np 
import torch 
from torch import nn 

def load_data():
  # 数据
  data = np.array([
    [1.0, 1.0],
    [2.0, 2.0],
    [3.0, 3.0],
    [4.0, 4.0],
    [5.0, 5.0],
    [6.0, 6.0],
    [7.0, 7.0],
    [8.0, 8.0],
    [9.0, 9.0],
    [10.0, 10.0],
    [11.0, 11.0],
    [12.0, 12.0],
    [13.0, 13.0],
    [14.0, 14.0],
    [15.0, 15.0],
    [0.0,0.0],
    [-1.0, 1.0],
    [-2.0, 2.0],
    [-3.0, 3.0],
    [-4.0, 4.0],
    [-5.0, 5.0],
    [-6.0, 6.0],
    [-7.0, 7.0],
    [-8.0, 8.0],
    [-9.0, 9.0],
    [-10.0, 10.0],
    [-11.0, 11.0],
    [-12.0, 12.0],
    [-13.0, 13.0],
    [-14.0, 14.0],
    [-15.0, 15.0]
  ])
  # 划分数据集，分训练集和测试集
  ratio = 0.8
  offset = int(data.shape[0] * ratio)
  train_data = data[:offset]
  test_data =  data[offset:]
  x_train_data = train_data[:, 0:1]
  y_train_data = train_data[:, 1:2]
  x_test_data = test_data[:, 0:1]
  y_test_data = test_data[:, 1:2]
  return x_train_data, y_train_data, x_test_data, y_test_data

x_train_data, y_train_data, x_test_data, y_test_data = load_data()

# 把数据转换为tensor张量
x_train_tensor = torch.tensor(x_train_data, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_data, dtype=torch.float32)

x_test_tensor = torch.tensor(x_test_data, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_data, dtype=torch.float32)

# 定义网络模型
model = nn.Sequential(
  # 创建线性模型
  # 第一个参数: 输入的特征
  # 第二个参数: 输出的目标
  nn.Linear(1,2),
  nn.ReLU(), #Relu激活函数
  nn.Linear(2,1)
)

# 定义损失函数
loss_fn = nn.MSELoss()

# 定义一个随机梯度下降优化器
optimizer = torch.optim.SGD(params=model.parameters(),lr=0.001)

# 开始训练
for i in range(3000):
  model.zero_grad() # 清楚原来的梯度信息
  y_pred = model(x_train_tensor) #前向计算
  # 计算损失
  loss = loss_fn(y_pred, y_train_tensor)
  # 计算梯度
  loss.backward()
  # 更新梯度
  optimizer.step()

  print(f'训练轮次:{i+1}, 具体的损失:{loss}')

# 打印模型的参数
print(model.state_dict())

# 模型评估
