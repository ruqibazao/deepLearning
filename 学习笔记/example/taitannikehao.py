import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# 泰坦尼克号数据集

# 加载数据
df = sns.load_dataset('titanic')
# 设置显示所有列 jupyter
pd.set_option('display.max_columns', None)

# print(df.head(30))

# 处理冗余数据
df.drop(columns=['class'], inplace=True)
df.drop(columns=['alive'], inplace=True)
df.drop(columns=['embarked'], inplace=True)

# print(df)

# 统计缺失情况
# print(df.isnull().sum())
# survived         0
# pclass           0
# sex              0
# age            177
# sibsp            0
# parch            0
# fare             0
# who              0
# adult_male       0
# deck           688
# embark_town      2
# alone            0
# dtype: int64

# 处理缺失值,使用均值填充 age ，列的缺失值
imputer = SimpleImputer(strategy='mean')
df['age'] = imputer.fit_transform(df[['age']])

# 处理缺失值 - 使用众数填充 embark_town 列的缺失值
# df['embark_town'].dropna()
# df['embark_town'].dtype 打印结果是 object，在 pandas 中，字符串被存储为 object 类型，这通常意味着这一列包含的是字符串数据
# 实际 embark_town（登陆港口全名） 是字符类型
# 这意味着您不能直接使用数值型数据的填充器（SimpleImputer）来填充
print(df['embark_town'].dtype)
# imputer3 = SimpleImputer(strategy='most_frequent')
# df['embark_town'] = imputer3.fit_transform(df[['embark_town']])

label_encoder = LabelEncoder()
df['embark_town'] = label_encoder.fit_transform(df['embark_town'])

# 处理缺失值 - 缺失太多，直接删除 deck 列
df.drop(columns=['deck'], inplace=True)

# 处理类别特证 - 使用 LabeEncoder 对 adult_male 进行整数编码
label_encoder = LabelEncoder()
df['adult_male'] = label_encoder.fit_transform(df['adult_male'])

# 处理类别特征 - 使用 LabelEncoder 对 alone（单独）进行整数编码
label_encoder = LabelEncoder()
df['alone'] = label_encoder.fit_transform(df['alone'])

# 处理类别特征 - 使用 OneHotEncoder 对 sex 进行独热编码
onehot_encoder = OneHotEncoder(sparse=False)
encoded_features = onehot_encoder.fit_transform(df[['sex']])
# 增加新的列： female, male
df = pd.concat([df, pd.DataFrame(encoded_features, columns=['female', 'male'])], axis=1)
# 把原来的 sex 删除
df.drop(columns=['sex'], inplace=True)
df['female'] = df['female'].astype(int)
df['male'] = df['male'].astype(int)

print(df)
