import numpy as np
import matplotlib.pyplot as plt
from sko.PSO import PSO
from matplotlib.animation import FuncAnimation
from predict_model import Pre_LinearRegression

coefficients, X_with_intercept = Pre_LinearRegression(
    "C:/Users/25749/Desktop/钻井数据/井A_工程参数深度数据 - 副本.xls")  # 系数向量
print("线性预测模型系数:", coefficients)
num_dimensions = len(coefficients) - 1
num_particles = X_with_intercept.shape[0]

# 设置搜索空间边界
X_train = X_with_intercept[:, 1:]
# 计算每个特征的最小值和最大值
lb = np.amin(X_train, axis=0)
ub = np.amax(X_train, axis=0)
#lb = np.zeros(num_dimensions)
#ub = np.amax(X_train * 10, axis=0)

def linear_function(x):
    """线性多元函数"""
    return np.dot(coefficients[1:], x) + coefficients[0]  # 线性函数的计算

constraint_ueq = (
    lambda x: 0.02-(np.dot(coefficients[1:], x) + coefficients[0])
    , lambda x: np.dot(coefficients[1:], x) + coefficients[0] - 10
)

max_iter = 150
pso = PSO(func=linear_function, dim=num_dimensions, pop=num_particles, max_iter=max_iter, lb=lb, ub=ub,constraint_ueq=constraint_ueq)
pso.record_mode = True
pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

plt.plot(pso.gbest_y_hist)
plt.show()

'''
# %% Now Plot the animation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

record_value = pso.record_value
X_list, V_list = record_value['X'], record_value['V']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('title', loc='center')

X_grid, Y_grid, Z_grid = np.meshgrid(np.linspace(-2.0, 2.0, 40), np.linspace(-2.0, 2.0, 40), np.linspace(-2.0, 2.0, 40))
Z_grid = linear_function((X_grid, Y_grid))
ax.contour3D(X_grid, Y_grid, Z_grid, 50)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)

t = np.linspace(0, 2 * np.pi, 100)
x = 0.5 * np.cos(t) + 1
y = 0.5 * np.sin(t)
z = np.zeros_like(x)
ax.plot3D(x, y, z, 'r')

p = plt.show()

scat = ax.scatter([], [], [], c='b', marker='.')

def update_scatter(frame):
    i, j = frame // 10, frame % 10
    ax.set_title('iter = ' + str(i))
    X_tmp = X_list[i] + V_list[i] * j / 10.0
    scat._offsets3d = (X_tmp[:, 0], X_tmp[:, 1], X_tmp[:, 2])
    return scat

ani = FuncAnimation(fig, update_scatter, blit=True, interval=25, frames=max_iter * 10)
plt.show()

ani.save('pso.gif', writer='pillow')
'''