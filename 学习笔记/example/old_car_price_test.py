import pandas as pd
import os

print("当前工作目录:", os.getcwd())

data = pd.read_csv('./example/train.csv')

# 删除所有空行数据
data.drop('id', axis=1, inplace=True)
data.dropna(inplace=True)
data.isnull().sum()

# 只取部分数据
# 品牌数据
engine_counts = data['brand'].value_counts()
top_20_engines = engine_counts.nlargest(20).index
data = data[data['brand'].isin(top_20_engines)]

# 型号数据
engine_counts = data['model'].value_counts()
top_20_engines = engine_counts.nlargest(20).index
data = data[data['model'].isin(top_20_engines)]

# fuel_type  汽车所使用的燃料类型
engine_counts = data['fuel_type'].value_counts()
top_20_engines = engine_counts.nlargest(10).index
data = data[data['fuel_type'].isin(top_20_engines)]

# engine
engine_counts = data['engine'].value_counts()
top_20_engines = engine_counts.nlargest(20).index
data = data[data['engine'].isin(top_20_engines)]

# transmission
engine_counts = data['transmission'].value_counts()
top_20_engines = engine_counts.nlargest(20).index
data = data[data['transmission'].isin(top_20_engines)]

# ext_col  
engine_counts = data['ext_col'].value_counts()
top_20_engines = engine_counts.nlargest(20).index
data = data[data['ext_col'].isin(top_20_engines)]

# int_col
engine_counts = data['int_col'].value_counts()
top_20_engines = engine_counts.nlargest(20).index
data = data[data['int_col'].isin(top_20_engines)]

# accident
engine_counts = data['accident'].value_counts()
top_20_engines = engine_counts.nlargest(20).index
data = data[data['accident'].isin(top_20_engines)]

# clean_title
engine_counts = data['clean_title'].value_counts()
top_20_engines = engine_counts.nlargest(20).index
data = data[data['clean_title'].isin(top_20_engines)]

# 把数据转换为one-hot编码
categorical_columns = ['brand', 'model', 'fuel_type', 'engine', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']
data = pd.get_dummies(data, columns=categorical_columns)
# 保存文件
# data.drop('Unnamed: 0', axis=1, inplace=True)
# data.to_csv('./example/train_03.csv', index=False)

print(data)

# 模型训练
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())

# 归一化处理
scaler = MinMaxScaler()
data[['price', 'model_year', 'milage']] = scaler.fit_transform(data[['price', 'model_year', 'milage']])

# 布尔类型的数据转换为 int 类型
categorical_cols = data.columns.difference(['price', 'model_year', 'milage'])
data[categorical_cols] = data[categorical_cols].astype('int')

# 分离特征和目录
input = data.drop('price', axis=1)
target = data['price']

# 转换为张量

input_tensor = torch.tensor(input.values, dtype=torch.float32)
target_tensor = torch.tensor(target.values, dtype=torch.float32).view(-1, 1)

# 定义模型
model = nn.Sequential(
    # 定义一个线性模型
    nn.Linear(124, 256),
    nn.ReLU(), # 非线性变化
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 1)
)

# 定义损失函数
loss_fn = nn.MSELoss()

# 定义梯度下降优化器
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)

# 训练模型
for i in range(1000):
    model.zero_grad()
    y_pred = model(input_tensor)
    loss = loss_fn(y_pred, target_tensor)
    loss.backward()
    optimizer.step()
    print("第{}次训练，损失为:{}".format(i+1, loss))
