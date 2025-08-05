# ============================ 데이터 셋 준비, 정규화, 전처리 ============================
import numpy as np
import pandas as pd
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)
pd.set_option('max_colwidth', 1000)
bostondf = pd.read_csv('BostonHousing.csv')
# print(bostondf)
# print(bostondf.info())
bostondf.dropna(axis=0, how='any', inplace=True)
print(bostondf)
print(bostondf.info())
print(bostondf.head(5))
# 입력 데이터 준비 ( 주택가격을 결정짓는 특성 데이터 )
housedata_input = bostondf[['crim','zn','nox','indus','age','dis','rad','tax','chas','rm','ptratio','b','lstat']].copy()
housedata_target = bostondf[['medv']] # 주택가격(타겟)

# 데이터셋 분리
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = \
    train_test_split(housedata_input, housedata_target, random_state=43)
# print(train_input.shape)
# print(test_input.shape)
# print(train_input[:5])
# print(train_target[:5])

train_input = train_input.to_numpy()  # 판단스 데이터프레임을 넘파이 배열로 변경
test_input = test_input.to_numpy()
train_target = train_target.to_numpy()
test_target = test_target.to_numpy()

# 추후에는 특성데이터의 정규화 ==> stardscaler()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_input)
test_scaled = scaler.transform(test_input)

# print(train_scaled.shape)
# print(test_scaled.shape)

# ================================================================================================================
# ============================ 모델 설계 (딥러닝) ============================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

LR_model = Sequential()
LR_model.add(Dense(units=30, input_dim=13, activation='relu'))
LR_model.add(Dense(units=6, activation='relu'))
LR_model.add(Dense(units=1, activation='linear'))

# LR_model.summary()

LR_model.compile(loss='mse', optimizer='adam', metrics=['mse'])

LR_model.fit(train_scaled, train_target, epochs=100, batch_size=10, verbose=1)

pre = LR_model.predict(test_scaled).flatten()
# print(pre)
# print(test_target[:1])

for i in range(10):
    print('실제가격 : {:.3f}, 예상가격 : {:.3f}'.format(test_target[i,0], pre[i]))
    # print(test_target[i], pre[i])