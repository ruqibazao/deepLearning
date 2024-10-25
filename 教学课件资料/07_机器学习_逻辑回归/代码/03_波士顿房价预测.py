from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1  加载波士顿房价数据集
boston = load_boston()
X = boston.data    # 特征数据
y = boston.target  # 目标数据

# 特征缩放 - Min-Max标准化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 创建线性回归模型
# model = LinearRegression()
model = SGDRegressor(learning_rate='constant', eta0=0.001, max_iter=50000)

# 在训练集上拟合模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 训练得到的数据: 权重和偏置
print(f"权重:{model.coef_},偏置:{model.intercept_}")

# 计算均方误差（Mean Squared Error）
mse = mean_squared_error(y_test, y_pred)
print("均方误差（MSE）：", mse)