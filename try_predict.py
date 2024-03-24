# 尝试对钻井数据进行标准规划
# from sklearn.preprocessing import MinMaxScaler
# import pandas as pd
# df = pd.read_excel("D:\Temp\drilling_date1.xlsx")
# # 假设df是包含钻井数据的pandas DataFrame
# # 选择需要进行最小-最大标准化的列
# columns_to_scale = ['x1', 'x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','y']  # 替换为实际的列名
#
# # 提取需要标准化的数据
# data_to_scale = df[columns_to_scale].values
#
# # 创建MinMaxScaler对象
# scaler = MinMaxScaler()
#
# # 对选定的列进行标准化
# df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
#
# # 将标准化后的数据保存为新的Excel文件
# df.to_excel('D:/Temp/scaled_drilling_data.xlsx', index=False, engine='openpyxl')

# 绘制相关性热力图
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # 假设你有一个名为'drilling_data.csv'的CSV文件，其中包含钻井数据
# df = pd.read_excel("D:\Temp\scaled_drilling_data.xlsx")
#
# # 计算皮尔逊相关系数矩阵
# corr_matrix = df.corr()
#
# # 创建一个热力图
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix,
#             xticklabels=corr_matrix.columns.values,
#             yticklabels=corr_matrix.columns.values,
#             annot=True,  # 显示数值
#             cmap='coolwarm',  # 使用冷暖色调的配色方案
#             linewidths=.5)  # 设置每个单元格之间的线宽
#
# plt.title('钻井数据皮尔逊相关系数热力图')
# plt.show()

# 利用随机森林算法计算数据的相关性(具有较强的可行度)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
# 加载钻井数据
df = pd.read_excel('D:\Temp\scaled_drilling_data.xlsx')  # 替换为你的数据文件路径

# 假设'target_column'是你要预测的目标列
X = df.drop('y', axis=1)
y = df['y']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化随机森林回归器
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
rf.fit(X_train, y_train)

# 获取特征重要性
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# 打印特征重要性
for f in range(X_train.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, X_train.columns[indices[f]], importances[indices[f]]))

# 可视化特征重要性
plt.figure(figsize=(12, 6))
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
        color="r", align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()

# # 利用随机森林算法计算数据的相关性,并且绘制热力图（x1存在一点小问题）
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# import seaborn as sns
# import matplotlib.pyplot as plt
# # 加载钻井数据
# df = pd.read_excel('D:\Temp\scaled_drilling_data.xlsx')  # 替换为你的数据文件路径
# # 以钻速'y'作为目标变量
# X = df.drop('y', axis=1)
# y = df['y']
# # 初始化随机森林回归器
# rf = RandomForestRegressor(n_estimators=100, random_state=42)
# # 训练模型
# rf.fit(X, y)
# # 获取特征重要性
# importances = rf.feature_importances_
# # 将特征重要性添加到DataFrame中
# feature_importances_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
# # 根据重要性对特征进行排序
# feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)
# # 绘制热力图
# plt.figure(figsize=(10, 8))
# sns.heatmap(feature_importances_df[['Feature', 'Importance']], annot=True, cmap='coolwarm', fmt=".2f")
# plt.xlabel('Features')
# plt.ylabel('Importance')
# plt.title('Feature Importances from Random Forest')
# plt.show()