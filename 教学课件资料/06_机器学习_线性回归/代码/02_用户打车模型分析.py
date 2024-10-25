# 1 导入基本模块
import numpy as np 
import plotly.graph_objects as go

# 2 加载数据 
# 训练数据(x_train_data, y_train_data)
# 测试数据(x_test_data, y_train_data)
# 函数: 参数, 返回值, 功能逻辑
def load_data(raito=0.8):
  # 数据 data.shape (15,2)
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
  # 把数据拆分为训练数据和测试数据, 比如我规定 训练数据 80%, 测试数据20%
  offset = int(data.shape[0]*raito)
  train_data = data[0:offset]
  test_data = data[offset:]
  x_train_data = train_data[:,0:1]
  y_train_data = train_data[:,1:]
  x_test_data = test_data[:,0:1]
  y_test_data = test_data[:,1:]
  return x_train_data,y_train_data,x_test_data,y_test_data

# 3 定义模型, 进行训练
class MyNet:

    # 初始化参数
    def __init__(self,w=0,b=0) :
        self.w = w
        self.b = b
        self.loss_value_list=[]
        self.w_value_list=[]
        self.b_value_list=[]

    # 预测结果函数
    def forward(self,x):
        # x: 代表训练数据的特征  向量 
        predict = self.w * x + self.b   # y: 预测结果, 也是一个向量
        return predict # 返回预测结果

    # 损失函数  均方误差  预测结果和实际结果的平方差的平均值
    def loss(self, predict, y):
        error = predict - y  # 向量之间的相减
        return np.mean(error**2)

    # 计算梯度 对损失进行求偏导, 分别计算w和b的梯度
    def gradient(self,x, y, z):
        # x: 训练数据的特征数据 距离
        # y: 训练数据的结果     实际费用
        # z: 预测结果:         预测费用
        dw = np.mean((z - y) * x)
        db = np.mean(z - y)
        return dw, db

    # 更新梯度  原来的w,b 需要跟新到最新的数据
    # 更新方式  w = w - dw * learn_rate
    # b = b - db*learn_rate
    def update(self, dw,db, learn_rate=0.01):
        self.w = self.w - dw * learn_rate
        self.b = self.b - db * learn_rate       

    # 训练方法
    def train(self,x,y,epoch=1000,learn_rate=0.01):
        
        # 5 打印一下日志信息
        # 0 循环训练轮次 比如说1000次
        for i in range(epoch):
            # 1 计算预测结果 predict  y=w*x+b
            predict = self.forward(x)  # 预测结果
            # 2 定义一个成本函数loss  均方误差
            loss_value = self.loss(predict,y)  
            self.loss_value_list.append(loss_value)
            self.w_value_list.append(self.w)
            self.b_value_list.append(self.b)
            # 3 计算梯度 分别对w, b 进行求导
            dw,db = self.gradient(x,y,predict)
            # 4 更新梯度   梯度*学习率  dw * learn_rate
            self.update(dw,db,learn_rate)
            print(f"训练轮次{i+1},损失值:{loss_value}, 更新后的w:{self.w},b:{self.b}")

        
if __name__=='__main__':
    x_train_data,y_train_data,x_test_data,y_test_data =load_data()
    # 开始模型训练 创建模型
    model = MyNet()
    # 开始训练
    model.train(x_train_data,y_train_data,epoch=5000,learn_rate=0.01)

    # 测试集的效果
    test_predict = model.forward(x_test_data)
    print('===========测试数据结果=============')
    print(test_predict)
    test_loss = model.loss(test_predict,y_test_data)
    print('测试数据误差==========')
    print(test_loss)

    # 创建损失值散点图
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(20,len(model.loss_value_list)+1)),
        y=model.loss_value_list[20:],
        mode='markers+lines',  # 同时绘制散点和折线
        marker=dict(color='blue', size=8),
        line=dict(color='blue', width=2),
        name='损失值'
    ))

    # 添加标题和轴标签
    fig.update_layout(
        title='线性模型训练过程中的损失值变化',
        xaxis_title='迭代次数',
        yaxis_title='损失值',
        template='plotly_white',
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    # 显示图形
    # fig.show()

    # 4. 绘制梯度下降路径的三维图形
    gd_path = go.Scatter3d(
        x=model.w_value_list,
        y=model.b_value_list,
        z=model.loss_value_list,
        mode='lines+markers',
        line=dict(color='blue', width=4),
        marker=dict(size=5, color='red'),
        name='梯度下降路径'
    )

    # 创建图形对象
    fig = go.Figure(data=[gd_path])

    # 更新布局
    fig.update_layout(
        title='梯度下降过程中的参数变化',
        scene=dict(
            xaxis_title='权重 w',
            yaxis_title='偏置 b',
            zaxis_title='损失 Loss'
        ),
        width=800,
        height=600
    )

    # 显示图形
    # fig.show()
    # 1 加载数据 训练数据(x_train_data, y_train_data) 测试数据(x_test_data, y_test_data)
    x = x_train_data
    y_true = y_train_data

    # 2. 定义损失函数
    def mse_loss(w, b, x, y_true):
        y_pred = w * x + b
        loss = np.mean((y_true - y_pred) ** 2)
        return loss

    # 3. 创建参数网格
    w_values = np.linspace(0, 12, 500)   # w 的取值范围
    b_values = np.linspace(0, 12, 500)   # b 的取值范围
    W, B = np.meshgrid(w_values, b_values)

    # 4. 计算损失值
    Loss = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Loss[i, j] = mse_loss(W[i, j], B[i, j], x, y_true)

    # 5. 绘制损失函数图像
    fig = go.Figure(data=[go.Surface(z=Loss, x=W, y=B)])
    fig.update_layout(title='均方误差（MSE）损失函数',
                    scene=dict(
                        xaxis_title='权重 w',
                        yaxis_title='偏置 b',
                        zaxis_title='损失 Loss'
                    ),
                    autosize=False,
                    width=800,
                    height=800)
    fig.show()