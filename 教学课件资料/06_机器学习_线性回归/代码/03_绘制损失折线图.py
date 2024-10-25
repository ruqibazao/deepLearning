import plotly.graph_objects as go

# 假设您已经有一个包含每次迭代损失值的列表或数组
loss_values = [0.9, 0.85, 0.8, 0.75, 0.73, 0.7, 0.68, 0.65, 0.63, 0.6]  # 示例数据

# 创建迭代次数的列表，长度与损失值列表相同
iterations = list(range(1, len(loss_values) + 1))

# 创建散点图
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=iterations,
    y=loss_values,
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
fig.show()