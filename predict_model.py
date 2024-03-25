import random
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler

from preprocess import Preprocess_RandomForest

# 训练线性回归预测模型
def Pre_LinearRegression(fpath):
    # 预处理
    X_selected, y = Preprocess_RandomForest(fpath)
    # 划分训练集和测试集
    #X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=random.randint(1, 1000))
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2,random_state=42)
    '''
    # 预测部分
    # 实例化 SGDRegressor 类，并设置参数,构建梯度下降优化的线性回归模型
    sgd_regressor = SGDRegressor(loss='squared_loss', learning_rate='constant', eta0=0.01, max_iter=1000,
                                 random_state=42)
    # 拟合模型
    sgd_regressor.fit(X_train, y_train)
    # 使用测试集进行预测
    y_pred = sgd_regressor.predict(X_test)
    # 计算均方误差进行模型评估
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    '''
    # 归一化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # 参数优化部分
    # y = w0+w1*x1+ ... +wn*xn 添加一列全为1的列作为截距项x0
    X_with_intercept_unscaled = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_with_intercept = np.c_[np.ones((X_train.shape[0], 1)), X_scaled]
    # 初始化模型参数
    theta = np.random.randn(X_with_intercept.shape[1])
    '''
    # 定义损失函数（均方误差）
    def mse_loss(X, y, theta):
        y_pred = np.dot(X, theta)
        return np.mean((y_pred - y) ** 2)
    '''
    # 定义梯度计算函数,公式grad(J(theta)) = (2 / m) * (y_pred - y) * X
    def compute_gradient(X, y, theta):
        y_pred = np.dot(X, theta)
        errors = y_pred - y # 残差
        gradient = 2 * np.dot(X.T, errors).astype(np.float64) / X.shape[0]
        return gradient # grad(J(theta))

    # 优化更新公式theta_i+1 = theta_i - learning_rate * gradient
    def optimize_theta(X_with_intercept, y_train, theta):
        # 设置梯度下降参数
        learning_rate = 1e-6  # 学习率
        max_iter = 1000  # 迭代次数
        tolerance = 1e-5
        # 梯度下降优化
        for i in range(max_iter):
            gradient = compute_gradient(X_with_intercept, y_train, theta)
            new_theta = theta - learning_rate * gradient
            # 判断是否收敛
            if np.linalg.norm(new_theta - theta) < tolerance:
                break
            theta = new_theta
        return theta
    optimized_theta = optimize_theta(X_with_intercept, y_train, theta)
    return optimized_theta, X_with_intercept_unscaled