# 加载数据
import pandas as pd
import torch
import torch.nn as nn 
from sklearn.preprocessing import MinMaxScaler

# 1 加载数据
data = pd.read_csv('./train_03.csv')

# 2 归一化处理
scaler = MinMaxScaler()
data[['price','model_year','milage']] = scaler.fit_transform(data[['price','model_year','milage']])

# 布尔类型的数据转换为int类型
categorical_cols = data.columns.difference(['price','model_year','milage'])
data[categorical_cols] = data[categorical_cols].astype(int)

# 分离特征和目标
input = data.drop('price',axis=1)
target = data['price']

# 转换为张量
input_tensor=torch.tensor(input.values,dtype=torch.float32)
target_tensor = torch.tensor(target.values,dtype=torch.float32).view(-1,1) # 转换为列向量

# 定义模型
model = nn.Sequential(
    # 定义一个线性模型
    # 输入特征数:124, 输出特征数:256
    nn.Linear(124,256),
    nn.ReLU(), #非线性变化
    # 线性模型
    nn.Linear(256,512),
    nn.ReLU(),
    # 定义最后一个输出层
    nn.Linear(512,1)
)

# 定义损失函数
loss_fn = nn.MSELoss()

# 定义一个梯度下降优化器
optimizer = torch.optim.SGD(params=model.parameters(),lr=0.001)

#开始模型训练
for i in range(1000):
    model.zero_grad()
    y_pred=model(input_tensor)
    loss = loss_fn(y_pred,target_tensor)
    loss.backward()
    optimizer.step()
    print(f'训练轮次:{i+1}, 损失值:{loss}')