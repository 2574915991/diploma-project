import pandas as pd

fpath = "C:/Users/25749/Desktop/钻井数据/井A_工程参数深度数据 - 副本.xls"
df = pd.read_excel(fpath)
df_interpolated = df.interpolate()
print("插值处理后的数据中的缺失值统计：")
print(df_interpolated.isnull().sum())
df_interpolated.to_excel("C:/Users/25749/Desktop/钻井数据/井A_工程参数深度数据 - handle.xlsx", index=False)