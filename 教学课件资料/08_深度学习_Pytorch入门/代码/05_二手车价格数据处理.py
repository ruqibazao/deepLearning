import pandas as pd 

# 读取文件
data = pd.read_csv('F:/nenhall/ing/python/example/train.csv')

# 1 删除id 列
data.drop('id',axis=1,inplace=True)
# 2 删除所有的空行数据
data.dropna(inplace=True)
data.isnull().sum()

# 3 只取部分数据
# 品牌数据
engine_counts = data['brand'].value_counts()
top_20_engines = engine_counts.nlargest(20).index
data = data[data['brand'].isin(top_20_engines)]

# 型号数据
engine_counts = data['model'].value_counts()
top_20_engines = engine_counts.nlargest(20).index
data = data[data['model'].isin(top_20_engines)]

# fuel_type 汽车所使用的燃料类型
engine_counts = data['fuel_type'].value_counts()
top_20_engines = engine_counts.nlargest(10).index
data = data[data['fuel_type'].isin(top_20_engines)]


# engine  
engine_counts = data['engine'].value_counts()
top_20_engines = engine_counts.nlargest(20).index
data = data[data['engine'].isin(top_20_engines)]

# transmission  
engine_counts = data['transmission'].value_counts()
top_20_engines = engine_counts.nlargest(20).index
data = data[data['transmission'].isin(top_20_engines)]

# ext_col  
engine_counts = data['ext_col'].value_counts()
top_20_engines = engine_counts.nlargest(20).index
data = data[data['ext_col'].isin(top_20_engines)]

# int_col 
engine_counts = data['int_col'].value_counts()
top_20_engines = engine_counts.nlargest(20).index
data = data[data['int_col'].isin(top_20_engines)]

# accident 
engine_counts = data['accident'].value_counts()
top_20_engines = engine_counts.nlargest(20).index
data = data[data['accident'].isin(top_20_engines)]

# clean_title 
engine_counts = data['clean_title'].value_counts()
top_20_engines = engine_counts.nlargest(20).index
data = data[data['clean_title'].isin(top_20_engines)]

# 把数据转换为one-hot编码
categorical_columns = ['brand', 'model', 'fuel_type','engine', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']
data = pd.get_dummies(data, columns=categorical_columns)

# 保存文件
# data.drop('Unnamed: 0',axis=1,inplace=True)
data.to_csv('train_03.csv',index=False)