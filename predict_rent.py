import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv('housing.csv')
sns.displot(df['x6'])

# 外れ値除去(3σ法)
col = 'x6'

#平均
mean = df.mean()
print(mean[col])

#標準偏差
sigma = df.std()
print(sigma[col])

#3σ
low = mean[col] - 3 * sigma[col] 
print(low)

high = mean[col] + 3 * sigma[col]
print(high)

df2 = df[(df[col] > low) & (df[col] < high)]
print(len(df2))

# 分布の確認
sns.displot(df2['x6'])

cols = df.columns
print(cols)

_df = df
for col in cols:
    # 3σ法の上下限値を設定
    low = mean[col] - 3 * sigma[col]
    high = mean[col] + 3 * sigma[col]
    #　条件での絞り込み
    _df = _df[(_df[col] > low) & (_df[col] < high)]

print(len(_df))

X = _df.iloc[:,:-1]
y = _df.iloc[:, -1]

# 訓練データと検証データに分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# 重回帰分析
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# scalerの学習　平均と標準偏差を計算
scaler.fit(X_train)

# scaling
X_train2 = scaler.transform(X_train)
X_test2 = scaler.transform(X_test)

model2 = LinearRegression()
model2.fit(X_train2, y_train)
print(model2.score(X_train2, y_train))

# 重みの確認
np.set_printoptions(precision=2, suppress=True)
print(model2.coef_)