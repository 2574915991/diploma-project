from sklearn.preprocessing import StandardScaler

from predict_model import Pre_LinearRegression
import numpy as np

# print(Pre_LinearRegression("C:/Users/25749/Desktop/钻井数据/井A_工程参数深度数据 - 副本.xls"))
coefficients, X_with_intercept = Pre_LinearRegression(
    "C:/Users/25749/Desktop/钻井数据/井A_工程参数深度数据 - 副本.xls")  # 系数向量
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
print("线性预测模型系数:", coefficients)


def linear_function(x, coefficient=coefficients):
    return np.dot(coefficient[1:], x) + coefficient[0]  # 线性函数的计算


def pso_optimizer_linear(linear_function, bounds, num_dimensions, num_particles, max_iter=200, w=0.5, c1=1.5, c2=1.5):
    # 初始化粒子群
    # particles = np.random.uniform(bounds[0], bounds[1], (num_particles, num_dimensions))
    particles = np.abs(np.random.rand(num_particles, num_dimensions))
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

    # 迭代优化
    for _ in range(max_iter):
        # 更新速度和位置
        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = w * velocities[i] + c1 * r1 * (personal_best_positions[i] - particles[i]) + \
                            c2 * r2 * (global_best_position - particles[i])
            particles[i] += velocities[i]
            particles[i] = np.maximum(particles[i], 0)  # 将负值设置为零

            # 确保粒子位置在边界内
            for d in range(num_dimensions):
                lower_bound = bounds[0, d]
                upper_bound = bounds[1, d]
                particles[i, d] = np.clip(particles[i, d], lower_bound, upper_bound)

        # 更新个体最佳位置和全局最佳位置
        values = np.array([linear_function(p) for p in particles])
        for i in range(num_particles):
            if values[i] < personal_best_values[i]:
                personal_best_positions[i] = particles[i]
                personal_best_values[i] = values[i]
        if np.min(personal_best_values) < global_best_value:
            global_best_index = np.argmin(personal_best_values)
            global_best_position = personal_best_positions[global_best_index]
            global_best_value = personal_best_values[global_best_index]
        '''
        global_best_position = np.maximum(global_best_position, 0)  # 将负值设置为零
        global_best_value = np.maximum(global_best_value, 0)
        '''
    return global_best_position, global_best_value


# 使用粒子群优化算法求解线性多元函数最小值
best_position, best_value = pso_optimizer_linear(linear_function, bounds, num_dimensions, num_particles)

print("最优解 x:", best_position)
print("最优值 f(x):", best_value)
