# Pytorch的基本概述

官网:https://pytorch.org/

文档: https://pytorch.org/docs/stable/index.html

快速开始: https://pytorch.org/get-started/locally/

## 历史背景

PyTorch 是一个开源的机器学习库，用于应用如计算机视觉和自然语言处理等领域的深度学习。它由Facebook的人工智能研究团队（FAIR）开发，并于2016年被公开发布。PyTorch的设计灵感部分来自于Torch，这是一个使用Lua语言编写的科学计算框架。不同于Torch，PyTorch 使用Python作为其开发语言，并利用了Python的丰富生态系统和易用性。

自发布以来，PyTorch迅速成长为最受欢迎的深度学习框架之一，与Google的TensorFlow竞争。它特别受到学术界和研究人员的喜爱，因为其简单的界面和灵活性，使得实验和原型设计更加直观和快速。它支持动态计算图（称为autograd系统），这是其区别于其他深度学习库的一个重要特征。动态计算图意味着图的行为可以在运行时改变，这对于实现复杂的模型和灵活的模型迭代具有重要意义。

随着时间的推移，PyTorch 推出了更多的高级功能，例如分布式训练、量化、模型导出（ONNX支持）等，使其在工业界也得到了广泛应用。PyTorch 还拥有一个活跃的社区，该社区为其库贡献了大量扩展和工具，如PyTorch Lightning、Hugging Face的Transformers 和 TorchVision等，进一步加强了其功能。

## 作用

PyTorch 在科学研究和工业应用中都发挥了重要作用。在学术界，它被广泛用于发表论文和教学中，由于其简洁性和灵活性，使得复杂的新思想可以快速实现并测试。在工业界，它被用于开发和部署面向生产的机器学习模型，支持各种从自动驾驶汽车到在线推荐系统的应用。

PyTorch 通过其简单的API、强大的功能以及与Python生态系统的紧密结合，在深度学习和AI领域中继续增强其重要性。不论是研究人员还是开发者，都可以从 PyTorch 强大的功能中受益，快速地从概念验证到全面部署。

## 基本功能

PyTorch是一个提供强大功能和灵活性的深度学习框架，被广泛用于研究和开发。以下是一些关于PyTorch的基本功能和主要特点的详细介绍：

### **张量计算（类似于NumPy）**

PyTorch 提供了一个高效的张量计算库，类似于 NumPy，但与之不同的是，PyTorch 张量可以利用 GPU 加速其数值计算。这使得数据科学家和研究人员能够利用强大的 GPU 计算能力，显著提高数据处理和模型训练的速度。

### **自动微分系统（Autograd）**

PyTorch 的一个核心特点是其动态计算图的自动微分系统，称为 `Autograd`。这个系统允许用户轻松计算所有参数的梯度，非常适合进行快速原型设计和复杂动态神经网络的实验。Autograd 自动记录你的代码执行的操作并在其上创建计算图，然后通过反向传播自动计算梯度。

###  **神经网络模块（torch.nn）**

PyTorch 通过 `torch.nn` 模块提供了构建深度神经网络所需的所有构建块。这包括各种预定义的层如全连接层、卷积层、激活层等，以及损失函数和优化器。这些工具是模块化的，可以轻松扩展和修改，提供了极大的灵活性和控制能力。

###  **模型训练和评估**

PyTorch 提供了方便的数据加载器 `torch.utils.data.DataLoader`，这使得加载和预处理大型数据集变得容易。此外，它还支持多种方式来监控训练过程，包括自定义训练循环、使用TensorBoard进行可视化等。PyTorch 还提供了模型的保存和加载机制，使得模型部署变得简单。

###  **分布式训练**

PyTorch 支持原生的分布式训练，允许数据科学家和工程师在多个 GPU 和服务器上并行处理数据和训练模型。这是通过 `torch.distributed` 包实现的，它提供了跨多个设备的数据并行能力。

###  **与其他语言和平台的兼容性**

PyTorch 支持与其他框架和平台的集成，例如与 ONNX（Open Neural Network Exchange）的兼容性，这使得在不同的框架之间迁移和部署模型变得可能。此外，它还支持与其他Python科学计算库（如 NumPy 和 SciPy）的无缝集成。

## 主要特点

- **灵活性和动态性**：PyTorch 的动态计算图使其在改变模型行为方面具有极大的灵活性，非常适合于研究和开发新算法。
- **易于学习和使用**：PyTorch 提供了直观的API，使得新手能够较容易地上手并开始构建深度学习模型。
- **社区支持**：作为一个受欢迎的开源项目，PyTorch 拥有一个非常活跃的社区，提供了大量的教程、工具和预训练模型，以及社区的支持。

PyTorch 不仅因其在研究领域的流行而受到青睐，也因其在工业应用中的有效性而被广泛采用。这些功能和特点使得 PyTorch 成为了执行复杂和创新深度学习项目的强大工具。

# 快速安装

对于刚入门, 建议先安装cpu版本, 对于少量的数据运行用不上gpu进行训练

这里我们直接安装最新版本即可

```python
pip install torch torchvision torchaudio
```

`先不用安装gpu版本`

验证是否安装成功

```python
import torch
print(torch.__version__)
```

# 案例实战

## 使用Pytorch完成用户打车案例

我们的用户打车案例就是一个简单的线性模型 y=wx+b

> ![image-20241020134218347](https://markdown-hesj.oss-cn-guangzhou.aliyuncs.com/image-20241020134218347.png)

数据:

| 打车距离 | 费用（元） |
| -------- | ---------- |
| 1.0      | 15.0       |
| 2.0      | 18.0       |
| 3.0      | 21.0       |
| 4.0      | 24.0       |
| 5.0      | 27.0       |
| 6.0      | 30.0       |
| 7.0      | 33.0       |
| 8.0      | 36.0       |
| 9.0      | 39.0       |
| 10.0     | 42.0       |
| 11.0     | 45.0       |
| 12.0     | 48.0       |
| 13.0     | 51.0       |
| 14.0     | 54.0       |
| 15.0     | 57.0       |

完整的实现代码

```python
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

# 将NumPy数组转换为PyTorch张量
x_train_tensor = torch.tensor(x_train_data, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_data, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test_data, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_data, dtype=torch.float32)

# 使用torch.nn.Sequential构建简单的线性模型
model = nn.Sequential(
    nn.Linear(1, 1)  # 输入和输出维度都是1，对应简单线性回归
)

# 使用均方误差(MSE)作为损失函数
loss_fn = nn.MSELoss()
# 使用随机梯度下降(SGD)优化器，设置学习率为0.01
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

# 训练模型
for i in range(5000):
    model.train()  # 将模型设置为训练模式
    optimizer.zero_grad()  # 清除之前的梯度
    y_pred = model(x_train_tensor)  # 对训练数据进行预测
    loss = loss_fn(y_pred, y_train_tensor)  # 计算损失
    loss.backward()  # 损失反向传播，计算梯度
    optimizer.step()  # 根据梯度更新模型参数
    if (i + 1) % 500 == 0:  # 每500次迭代输出一次损失值
        print(f"训练轮次: {i + 1}, 当前的损失值: {loss.item()}")
# 打印模型参数，检查学习到的权重和偏置
print("Model parameters:", model.state_dict())
for name, param in model.named_parameters():
    print(f"{name}: {param}")
# 切换到评估模式
model.eval()
# 不计算梯度，进行模型评估
with torch.no_grad():
    y_pred = model(x_test_tensor)
    eval_result = loss_fn(y_pred, y_test_tensor)
    print(f'测试数据集评估结果: {eval_result.item()}')
```

## 使用Pytorch完成非线性关系的数据处理

数据:

| 特征 | 输出 |
| ---- | ---- |
| -10  | 10   |
| -9   | 9    |
| -8   | 8    |
| -7   | 7    |
| -6   | 6    |
| -5   | 5    |
| -4   | 4    |
| -3   | 3    |
| -2   | 2    |
| -1   | 1    |
| 0    | 0    |
| 1    | 1    |
| 2    | 2    |
| 3    | 3    |
| 4    | 4    |
| 5    | 5    |
| 6    | 6    |
| 7    | 7    |
| 8    | 8    |
| 9    | 9    |
| 10   | 10   |

数据特点:

1. 特征和输出之间不在是线性关系, 是`非线性关系`
2. 怎么使用线性关系表示非线性关系
3. 数据的函数图像是 y=|x|

> ![image-20241020141920992](https://markdown-hesj.oss-cn-guangzhou.aliyuncs.com/image-20241020141920992.png)

> ![image-20241020141951872](https://markdown-hesj.oss-cn-guangzhou.aliyuncs.com/image-20241020141951872.png)

> ![image-20241020142018175](https://markdown-hesj.oss-cn-guangzhou.aliyuncs.com/image-20241020142018175.png)

> ![image-20241020142213110](https://markdown-hesj.oss-cn-guangzhou.aliyuncs.com/image-20241020142213110.png)

> ![image-20241020142258832](https://markdown-hesj.oss-cn-guangzhou.aliyuncs.com/image-20241020142258832.png)

> relu激活函数: relu(x)=max(0,x)

如果要完成上面的函数图形:

1. 定义一个函数 y1 = x
2. 使用激活函数relu 对y1 进行处理  r1 = relu(y1)
3. 定义一个函数 y2 = -x
4. 使用激活函数relu 对y2 进行处理 r2 = relu(y2)
5. 定义一个函数输出 o = r1+r2

通过上面的处理, 就可以完成对应的函数图形

> ![image-20241020143517986](https://markdown-hesj.oss-cn-guangzhou.aliyuncs.com/image-20241020143517986.png)

其中网络模型

> ![image-20241020144105033](https://markdown-hesj.oss-cn-guangzhou.aliyuncs.com/image-20241020144105033.png)

完整代码实现

```python
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


# 加载数据
x_train_data, y_train_data, x_test_data, y_test_data = load_data()

# 把数据转换为Tensor的处理对象
x_train_tensor = torch.tensor(x_train_data,dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_data, dtype=torch.float32)

x_test_tensor = torch.tensor(x_test_data,dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_data, dtype=torch.float32)

# 定义网络模型 
model = nn.Sequential(
    # 定义一层网络模型
    nn.Linear(1,2),
    nn.ReLU(),
    nn.Linear(2,1)
)

# 定义损失函数
loss_fn = nn.MSELoss()
# 定义优化器配置
optimizer = torch.optim.SGD(params=model.parameters(),lr=0.001)

# 开始模型训练
for i in range(50000):
    model.zero_grad() #清除原来的梯度
    y_pred = model(x_train_tensor)
    loss = loss_fn(y_pred,y_train_tensor) # 计算损失
    loss.backward() #反向传播, 计算梯度
    optimizer.step() # 更新权重参数
    if (i+1)%100==0:
       print(f'训练轮次:{i+1}, 当前损失:{loss}')

# 训练后的模型参数
print(f'模型参数{model.state_dict()}')
for name,param in model.named_parameters():
   print(f'{name}:{param}')

# 模型评估
with torch.no_grad():
   y_pred = model(x_test_tensor)
   eval_result = loss_fn(y_pred,y_test_tensor)
   print(f'测试集评估结果:{eval_result.item()}')
```

## 案例: 二手车价格预测

### 数据说明

| 字段                       | 说明                                               |
| -------------------------- | -------------------------------------------------- |
| Brand                      | 品牌                                               |
| Model                      | 具体型号                                           |
| Model Year                 | 汽车的制造年份                                     |
| Mileage                    | 汽车的行驶里程（磨损程度和潜在维护需求的关键指标） |
| Fuel Type                  | 汽车所使用的燃料类型（汽油、柴油、电动、混合动力） |
| Engine Type                | 发动机规格                                         |
| Transmission               | 变速器类型，（自动挡、手动挡、其他）               |
| Exterior & Interior Colors | 外观颜色                                           |
| Exterior & Interior Colors | 内饰颜色                                           |
| Accident History           | 车辆是否有事故或损坏的历史记录                     |
| Clean Title                | 是否拥有健全良好的所有权证明                       |
| Price                      | 汽车标价                                           |

### 数据基本处理

```python
import pandas as pd 

# 读取文件
data = pd.read_csv('train.csv')

# 1 删除id 列
data.drop('id',axis=1,inplace=True)
# 2 删除所有的空行数据
data.dropna(inplace=True)
data.isnull().sum()

# 3 只取部分数据
# 品牌数据
engine_counts = data['brand'].value_counts()
top_20_engines = engine_counts.nlargest(20).index
data = data[data['brand'].isin(top_20_engines)]

# 型号数据
engine_counts = data['model'].value_counts()
top_20_engines = engine_counts.nlargest(20).index
data = data[data['model'].isin(top_20_engines)]

# fuel_type 汽车所使用的燃料类型
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
categorical_columns = ['brand', 'model', 'fuel_type','engine', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']
data = pd.get_dummies(data, columns=categorical_columns)
# 保存文件
data.drop('Unnamed: 0',axis=1,inplace=True)
data.to_csv('train_03.csv',index=False)
```

### 模型训练代码

```python
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
target_tensor = torch.tensor(target.values,dtype=torch.float32).view(-1,1)

# 定义模型
model = nn.Sequential(
    nn.Linear(124,256),
    nn.ReLU(),
    nn.Linear(256,128),
    nn.ReLU(),
    nn.Linear(128,1)
)

# 定义损失函数
loss_fn = nn.MSELoss()
# 定义优化器
optimizer = torch.optim.Adam(params=model.parameters(),lr=0.001)

# 开始模型训练 
for i in range(1000):
    model.zero_grad()
    y_pred = model(input_tensor)
    loss = loss_fn(y_pred,target_tensor)
    loss.backward()
    optimizer.step()
    print(f'训练轮次:{i+1}, 损失:{loss}')
```

# 作业

## 作业1 完成波士顿房价的预测

数据加载: 直接从sklearn中加载默认的数据集

使用神经网络进行模型训练

优化器使用SGD

损失函数使用均方误差 MSE

自定义网络模型

## 作业2 使用google的colab完成上述代码的编写和运行

https://colab.research.google.com/

需要注册google邮箱

需要科学上网

