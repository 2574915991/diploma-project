from predict_model import Pre_LinearRegression
import numpy as np

#print(Pre_LinearRegression("C:/Users/25749/Desktop/钻井数据/井A_工程参数深度数据 - 副本.xls"))
coefficients, X_with_intercept = Pre_LinearRegression("C:/Users/25749/Desktop/钻井数据/井A_工程参数深度数据 - 副本.xls")  # 系数向量
num_dimensions = len(coefficients) - 1
num_particles = X_with_intercept.shape[0]

# 设置搜索空间边界
X_train = X_with_intercept[:, 1:]
# 计算每个特征的最小值和最大值
min_values = np.amin(X_train, axis=0)
max_values = np.amax(X_train, axis=0)
# 将最小值和最大值组合成一个二维数组
bounds = np.vstack((min_values, max_values))
'''
print(min_values)
print(max_values)
print(bounds)
'''
def linear_function(coefficients, X_with_intercept):
    return np.dot(coefficients, X_with_intercept) # 线性函数的计算
def pso_optimizer_linear(linear_function, bounds, num_dimensions, num_particles, max_iter=100, w=0.5, c1=1.5, c2=1.5):
    # 初始化粒子群
    #particles = np.random.uniform(bounds[0], bounds[1], (num_particles, num_dimensions))
    particles = np.empty((num_particles, num_dimensions))
    for d in range(num_dimensions):
        lower_bound = bounds[0][d]
        upper_bound = bounds[1][d]
        particles[:, d] = np.random.uniform(lower_bound, upper_bound, size=num_particles)  # 初始化当前特征的粒子位置
    velocities = np.zeros((num_particles, num_dimensions))
    personal_best_positions = particles.copy()
    personal_best_values = np.array([linear_function(p) for p in personal_best_positions])
    global_best_index = np.argmin(personal_best_values)
    global_best_position = personal_best_positions[global_best_index]
    global_best_value = personal_best_values[global_best_index]

