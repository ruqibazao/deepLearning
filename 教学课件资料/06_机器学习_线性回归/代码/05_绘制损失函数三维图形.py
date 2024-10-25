# 导入基本模块
import numpy as np
import plotly.graph_objects as go

# 加载数据的函数load
# 1 加载数据 训练数据(x_train_data, y_train_data) 测试数据(x_test_data, y_test_data)
def load(ratio=0.8):
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
    offset = int(data.shape[0]*0.8)
    train_data=data[:offset]
    test_data = data[offset:]
    x_train_data = train_data[:,:1]
    y_train_data = train_data[:,1:]
    x_test_data = test_data[:,:1]
    y_test_data = test_data[:,1:]
    return x_train_data,y_train_data, x_test_data, y_test_data

# 定义一个网络模型类 MyNet
class MyNet:
    # 初始化函数
    def __init__(self, w=0, b=0) -> None:
        self.w = w
        self.b = b
        self.loss_values=[]
        self.loss_info_list=[]

    # 3 定义模型, 计算预测结果 y = w*x + b
    def predict(self, x):
        y = self.w * x + self.b 
        return y
    
    # 4 进行评价, 定义一个损失函数
    def loss(self, y, z):
        error = z-y
        return np.mean(error*error)
    
    # 5 计算梯度
    def gradient(self,x,y,z):
        dw = np.mean((z-y)*x)
        db = np.mean(z-y)
        return dw,db
    
    # 6 更新梯度
    def update(self, dw,db,learning_rate=0.01):
        self.w = self.w - learning_rate*dw
        self.b = self.b - learning_rate*db

    # 1 定义一个模型训练的方法 train
    def train(self,x_train_data,y_train_data,epoch=2000,learning_rate=0.01):
        for i in range(epoch):
            # 预测结果
            y_predict = self.predict(x_train_data)
            # 计算损失
            loss_value = self.loss(y_train_data,y_predict)
            # 计算梯度
            dw,db = self.gradient(x_train_data,y_train_data,y_predict)
            # 记录对应的loss, w, b 的对应数据
            self.loss_info_list.append((self.w,self.b,loss_value))
            # 更新梯度
            self.update(dw,db,learning_rate)
            # 把所有的损失都记录起来
            self.loss_values.append(loss_value)
            print(f'训练轮次:{i+1}, 当前的损失:{loss_value}, 更新后的w:{self.w}, b:{self.b}')

# 测试代码
if __name__=="__main__":
    x_train_data,y_train_data, x_test_data, y_test_data = load()
    # 创建一个实例对象
    model = MyNet(w=0,b=0)
    model.train(x_train_data,y_train_data,epoch=5000)

    # 1 绘制一个损失值的折线图
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1,len(model.loss_values)+1)),
        y=model.loss_values,
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

    # 2 绘制一个梯度下降的函数过程图像
    gd_path = go.Scatter3d(
        x=[value[0] for value in model.loss_info_list],
        y=[value[1] for value in model.loss_info_list],
        z=[value[2] for value in model.loss_info_list],
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

    # 3 函数损失图结合梯度下降图
    # 2. 定义损失函数
    def mse_loss(w, b, x, y_true):
        y_pred = w * x + b
        loss = np.mean((y_true - y_pred) ** 2)
        return loss
    # 打印w, b的取值范围
    w_min=np.min([value[0] for value in model.loss_info_list])
    w_max=np.max([value[0] for value in model.loss_info_list])
    b_min=np.min([value[1] for value in model.loss_info_list])
    b_max=np.max([value[1] for value in model.loss_info_list])
    # 3. 创建参数网格
    w_values = np.linspace(w_min-1, w_max+1, 500)   # w 的取值范围
    b_values = np.linspace(b_min-1, b_max+1, 500)   # b 的取值范围
    W, B = np.meshgrid(w_values, b_values)
    # 4. 计算损失值
    Loss = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Loss[i, j] = mse_loss(W[i, j], B[i, j], x_train_data, y_train_data)
    surface = go.Surface(
        x=W,
        y=B,
        z=Loss,
        colorscale='Viridis',
        opacity=0.8,
        name='损失函数曲面'
    )

    # 创建梯度下降路径的线条
    gd_path = go.Scatter3d(
        x=[value[0] for value in model.loss_info_list],
        y=[value[1] for value in model.loss_info_list],
        z=[value[2] for value in model.loss_info_list],
        mode='lines+markers',
        line=dict(color='red', width=4),
        marker=dict(size=5, color='red'),
        name='梯度下降路径'
    )

    # 创建图形对象
    fig = go.Figure(data=[surface, gd_path])

    # 更新布局
    fig.update_layout(
        title='梯度下降过程中的损失函数变化',
        scene=dict(
            xaxis_title='权重 w',
            yaxis_title='偏置 b',
            zaxis_title='损失 Loss'
        ),
        width=800,
        height=600
    )
    # 显示图形
    fig.show()