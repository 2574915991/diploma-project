import pandas as pd

path1 = "C:/Users/25749/Desktop/钻井数据/井A_工程参数深度数据 - handle.xlsx"
path2 = "C:/Users/25749/Desktop/钻井数据/井A_工程参数深度数据 - 副本.xls"

df1 = pd.read_excel(path1)
df2 = pd.read_excel(path2)

row_diff = df1.compare(df2, align_axis=0, keep_shape=True)

# 逐列比较
column_diff = df1.compare(df2, align_axis=1, keep_shape=True)

# 输出行差异
print("行差异:")
print(row_diff)

# 输出列差异
print("列差异:")
print(column_diff)

# 将行差异保存为 Excel 文件
row_diff.to_excel("C:/Users/25749/Desktop/钻井数据/row_diff.xlsx")

# 将列差异保存为 Excel 文件
column_diff.to_excel("C:/Users/25749/Desktop/钻井数据/column_diff.xlsx")