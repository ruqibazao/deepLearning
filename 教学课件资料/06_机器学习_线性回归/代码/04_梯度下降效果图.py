import numpy as np
import plotly.graph_objects as go

# 1. 定义数据集
x = np.array([1, 2, 3, 4, 5])
y_true = np.array([2, 4, 6, 8, 10])

# 2. 定义损失函数和梯度计算
def mse_loss(w, b, x, y_true):
    y_pred = w * x + b
    loss = np.mean((y_true - y_pred) ** 2)
    return loss

def compute_gradients(w, b, x, y_true):
    N = len(x)
    y_pred = w * x + b
    dw = (-2 / N) * np.sum(x * (y_true - y_pred))
    db = (-2 / N) * np.sum(y_true - y_pred)
    return dw, db

# 3. 实现梯度下降算法
w = 0.0  # 初始化权重
b = 0.0  # 初始化偏置
learning_rate = 0.01
num_iterations = 100

# 记录梯度下降的路径
w_history = []
b_history = []
loss_history = []

for _ in range(num_iterations):
    loss = mse_loss(w, b, x, y_true)
    dw, db = compute_gradients(w, b, x, y_true)
    w -= learning_rate * dw
    b -= learning_rate * db
    w_history.append(w)
    b_history.append(b)
    loss_history.append(loss)

# 4. 绘制梯度下降路径的三维图形
gd_path = go.Scatter3d(
    x=w_history,
    y=b_history,
    z=loss_history,
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
fig.show()