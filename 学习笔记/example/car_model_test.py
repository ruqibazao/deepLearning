# 用户打车模型分析
# import pytorch
import numpy as np
import plotly.graph_objects as go

# print(pytorch.__version__)


# 加载数据
# 训练数据
# 测试数据
# 函数：参数，返回值，功能逻辑
def load_data(rate=0.8):
    # 数据 [公里数，预测费用]
    data = np.array(
        [
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
            [15.0, 57.0],
        ]
    )
    # 把数据拆分为训练数据和测试数据，比如：80%训练，20%测试
    offset = int(data.shape[0] * rate)
    train_data = data[0:offset]
    test_data = data[offset:]
    x_train_data = train_data[0:, 0:1]
    y_train_data = train_data[:, 1:]
    x_test_data = test_data[:, 0:1]
    y_test_data = test_data[:, 1:]
    return x_train_data, y_train_data, x_test_data, y_test_data


# 定义模型，进行训练
class MyNet:

    # 初始化参数
    def __init__(self, w=0, b=0):
        self.w = w
        self.b = b
        self.loss_list = []
        self.w_value_list = []
        self.b_value_list = []
        self.loss_info_list = []

    # 预测结果函数
    def forward(self, x):
        # x: 代表训练数据的特征 向量
        predict = self.w * x + self.b  # predict 预测结果，也是一个向量
        return predict

    # 损失函数, 均方误差，预测结果和实际结果的平方差的平均值
    def loss(self, predict, y):
        # 向量之间的相减
        error = predict - y
        return np.mean(error**2)

    # 计算梯度
    def gradient(self, x, y, z):
        # x: 训练数据的特征向量，距离
        # y: 训练数据的结果， 实际费用
        # z: 预测结果，预测费用
        dw = np.mean((z - y) * x)
        db = np.mean(z - y)
        return dw, db

    # 更新梯度，原来的w，b 需要更新到时最新的数据
    # 更新方式 w = w - dw * learn_rate
    # b = b - db * learn
    def update(self, dw, db, learn_rate=0.01):
        self.w = self.w - dw * learn_rate
        self.b = self.b - db * learn_rate

    def train(self, x, y, epoch=1000, learn_rate=0.01):
        # 1. 循环训练轮次，例如：1000次
        # 2. 计算预测结果 predict y=w*x+b
        # 3. 定义一个成本函数 loss，均方误差
        # 4. 计算梯度，分别对w和b求导
        # 5. 更新梯度，梯度 * 学习率 （dw * learning_rate）
        for i in range(epoch):
            predict = self.forward(x)
            # print(predict), print(y)
            loss_value = self.loss(predict, y)
            self.loss_list.append(loss_value)
            self.w_value_list.append(self.w)
            self.b_value_list.append(self.b)
            dw, db = self.gradient(x, y, predict)
            # 记录对应的loss, w, b 的对应数据
            self.loss_info_list.append((self.w, self.b, loss_value))
            self.update(dw, db, learn_rate)
            print(
                f"训练轮次{i+1}，损失值:{loss_value}, 更新后的w:{self.w}, 更新后的b:{self.b}"
            )


# 画散点图
def show_scatter(model: MyNet):
    # 创建损失值散点图
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(model.loss_list) + 1)),
            y=model.loss_list,
            mode="lines+markers",
            marker=dict(color="blue", size=8),
            line=dict(color="blue", width=2),
            name="损失值",
        )
    )
    # 添加标题和轴标签
    fig.update_layout(
        title="线性模型训练过程中的损失值变化",
        xaxis_title="迭代次数",
        yaxis_title="损失值",
        template="plotly_white",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
    )
    # 显示图形
    fig.show()


# 画3D图
def show_scatter3D(model: MyNet):
    # 画 3D 图
    gd_path = go.Scatter3d(
        x=model.w_value_list,
        y=model.b_value_list,
        z=model.loss_list,
        mode="lines+markers",
        line=dict(color="blue", width=4),
        marker=dict(size=5, color="blue"),
        name="梯度下降路径",
    )
    fig = go.Figure(data=[gd_path])
    fig.update_layout(
        title="梯度下降过程中的参数变化",
        scene=dict(xaxis_title="权重 w", yaxis_title="偏置 b", zaxis_title="损失 Loss"),
        width=800,
        height=600,
    )
    # 显示图形
    fig.show()


# 定义损失函数
def mse_loss(w, b, x, y_true):
    y_pred = w * x + b
    loss = np.mean((y_pred - y_true) ** 2)
    return loss


# 画梯度图
def show_gradient(model: MyNet):
    # 函数损失图，结合梯度下降图
    w_min = np.min([value[0] for value in model.loss_info_list])
    w_max = np.max([value[0] for value in model.loss_info_list])
    b_min = np.min([value[1] for value in model.loss_info_list])
    b_max = np.max([value[1] for value in model.loss_info_list])

    # 创建参数网格线
    w_value = np.linspace(w_min - 1, w_max + 1, 500)  # w 的取值范围
    b_value = np.linspace(b_min - 1, b_max + 1, 500)  # b 的取值范围
    W, B = np.meshgrid(w_value, b_value)
    # 计算损失值
    Loss = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Loss[i, j] = mse_loss(W[i, j], B[i, j], x_train_data, y_train_data)

    surface = go.Surface(
        x=W, y=B, z=Loss, colorscale="Viridis", opacity=0.8, name="损失函数曲面"
    )
    gd_path = go.Scatter3d(
        x=[value[0] for value in model.loss_info_list], # 权重数据列表
        y=[value[1] for value in model.loss_info_list], # 偏置数据列表
        z=[value[2] for value in model.loss_info_list], # 损失数据列表
        mode="lines+markers",
        line=dict(color="red", width=4),
        marker=dict(size=5, color="red"),
        name="梯度下降路径",
    )
    # 创建图形
    fig = go.Figure(data=[surface, gd_path])
    # 更新布局
    fig.update_layout(
        title="梯度下降过程中的参数变化",
        scene=dict(xaxis_title="权重 w", yaxis_title="偏置 b", zaxis_title="损失 Loss"),
        width=800,
        height=600,
    )
    # 显示图形
    fig.show()


if __name__ == "__main__":
    print("This is the main function")
    x_train_data, y_train_data, x_test_data, y_test_data = load_data(rate=0.8)
    # 创建模型
    model = MyNet(w=100, b=-1000)
    model.train(x_train_data, y_train_data, epoch=5000, learn_rate=0.01)

    # 测试集的效果
    test_predict = model.forward(x_test_data)
    print(f"测试集的预测结果：{test_predict}")
    test_loss = model.loss(test_predict, y_test_data)
    print(f"测试集的损失值：{test_loss}")

    numbers = np.linspace(1, 10, 20)
    print(numbers)

    # 画图
    # show_scatter(model=model)
    # show_scatter3D(model=model)
    show_gradient(model=model)
