import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# preprocess为预处理函数,fpath为待预处理文件路径
def Preprocess_RandomForest(fpath):
    # 缺失值处理
    #fpath = "C:/Users/25749/Desktop/钻井数据/井A_工程参数深度数据 - 副本.xls"
    df = pd.read_excel(fpath)
    df_interpolated = df.interpolate()
    '''
    print("插值处理后的数据中的缺失值统计：")
    print(df_interpolated.isnull().sum())
    df_interpolated.to_excel("C:/Users/25749/Desktop/钻井数据/井A_工程参数深度数据 - handle.xlsx", index=False)
    '''

    # 提取特征和目标变量
    X = df_interpolated.drop(columns=['钻时(min/m)', '套压(Mpa)', '转盘转速(r/min)', '纯钻时间h'])
    y = df_interpolated['钻时(min/m)']

    # 输出特征名称索引
    feature_index_mapping = {index: feature for index, feature in enumerate(X.columns)}
    '''
    for index, feature in feature_index_mapping.items():
        print(f"Index: {index}, Feature: {feature}")
    '''
    feature_list = list(feature_index_mapping.items())

    # 建立模型
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=random.randint(1, 1000))

    # 训练模型
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=random.randint(1, 1000))  # 划分训练集和测试集
    rf_regressor.fit(X_train, y_train)

    '''
    # 评估模型
    y_pred = rf_regressor.predict(X_test) # 对目标值预测
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse) # 用均方误差来评估
    '''

    # 特征重要性分析
    feature_importances = rf_regressor.feature_importances_
    '''
    for index, feature in enumerate(feature_importances):
        print(f"Index: {feature_list[index]}, Feature_importance: {feature}")
    '''
    # 选择重要性大于阈值的特征
    threshold = np.mean(feature_importances)  # 阈值设置为平均值
    # threshold = 0.05 # 自定义阈值
    print("threshold:", threshold)
    selected_features_indexs = [index for index, importance in enumerate(feature_importances) if importance > threshold]
    '''
    for index in selected_features_indexs:
        print(
            f"Select_Feature Index:{index} Feature_name: {feature_list[index]} Feature_importance: {feature_list[index]}")
    '''
    X_selected = X[X.columns[selected_features_indexs].tolist()]
    #print(X_selected)
    return X_selected, y # 返回挑选的X, y
'''
X_selected, y = Preprocess_RandomForest("C:/Users/25749/Desktop/钻井数据/井A_工程参数深度数据 - 副本.xls")
print(X_selected)
print(y)
'''