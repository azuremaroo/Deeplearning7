import tensorflow.keras as keras
import pandas as pd
import numpy as np
import csv

# multi regression : 여러 개의 독립 변수(입력 특징)를 사용하여
# 하나의 종속 변수(출력 값)를 예측하는 모델

def make_xy_1():
    f = open('trees.csv', encoding='utf-8')
    f.readline()
    x, y = [], []
    for _, girth, height, volume in csv.reader(f): # csv.reader(f, delimiter=',') : 각 열을 리스트로 반환
        x.append([float(girth), float(height)])
        y.append([float(volume)])

    f.close()
    return np.float32(x), np.float32(y) # np.float32() : numpy 배열을 만들 때 데이터를 모두 float32 로 변환하는 함수

# x, y = make_xy_1()
# print(x.shape, y.shape)
# print(x)
# print(y)
# exit()

def make_xy_2():
    trees = pd.read_csv('trees.csv', index_col=0)
    # trees = pd.read_csv('trees.csv', index_col=1, skiprows=0, usecols=[0, 1, 2, 3])
    # trees = pd.read_csv('trees.csv', index_col='Height', usecols=['Girth', 'Height', 'Volume'])

    # print(trees.info())
    # DataFrame : index, columns, values
    # print('index : ', trees.index)
    # print('columns : ', trees.columns)
    # print('values : ', trees.values)
    # print(trees.Girth)

    x = np.transpose([trees.Girth.values, trees.Height.values])  # 전치 행렬
    y = np.reshape(trees.Volume.values, [-1, 1])  # 학습 데이터 포맷으로 변경
    return x, y

x, y = make_xy_2()
# print(x[:5])
# print(y[:5])
# print(x.shape)
# print(y.shape)
# exit()

model = keras.Sequential()
model.add(layer=keras.layers.Dense(1))
model.compile(
    optimizer=keras.optimizers.SGD(.0001),
    loss=keras.losses.mean_squared_error
)
model.fit(x, y, epochs=100) # 데이터 질이 안좋을수록 학습 횟수를 늘려야 함

# # 퀴즈 : Girth 가 10, 20 Height 가 70, 80 일 때 Volume 구하기
x_test = np.array([[10., 70.], [20., 80.]])
print(model.predict(x_test))


