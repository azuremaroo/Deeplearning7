import numpy as np, pandas as pd
import tensorflow.keras as keras
from sklearn import preprocessing, model_selection

np.set_printoptions(threshold=np.inf) #무한으로 출력합니다. (sys.maxsize 크기 만큼 출력
np.set_printoptions(suppress=True) # 소숫점의 과학적 표기법의 사용 억제

def make_xy():
    pima = pd.read_csv('diabetes.csv')
    print(pima.info())
    # print(pima.head())
    x = pima.values[:, :-1] # 모든 행, 마지막 열을 제외한 나머지 모든 열
    y = pima.values[:, -1:] # 모든 행, 마지막 열부터 끝까지(마지막 열만)
    # print(x.dtype, y.dtype) # float64 float64

    return x, y

x, y = make_xy()
# print(x.shape, y.shape) # (768, 8) (768, 1)

# 표준화 : 데이터의 분포를 평균 0, 표준편차 1인 표준 정규 분포로 변환하는 방식(학습하기 좋은 데이터로 변환)
# x = preprocessing.scale(x) # 표준화하고 상태를 저장하지 않음
# x = preprocessing.minmax_scale(x)

scaler = preprocessing.StandardScaler() # 표준화하고 상태를 저장 ==> 모델 학습에 적합
# x = scaler.fit_transform(x)
# print(x.min(), x.max())
# print(scaler.scale_) # 저장된 상태 보는 방법
# print(scaler.mean_)

# print('len(x) : ', len(x)) # 760
# train_size = int(len(x) * .7) # 학습 자료의 70퍼
# print('train_size : ', train_size) # 532
# x_train, x_test = x[:train_size], x[train_size:]
# y_train, y_test = y[:train_size], y[train_size:]
data = model_selection.train_test_split(x, y, train_size=.7)
x_train, x_test, y_train, y_test = data

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) # transform() : 위에서 학습한 평균과 표준편차를 그대로 적용함

model = keras.Sequential()
model.add(keras.layers.Dense(1, activation='sigmoid')) # 행렬 곱셈 후 sigmoid 로 값을 0 또는 1로 반환

model.compile(
    optimizer=keras.optimizers.SGD(.1),
    loss=keras.losses.binary_crossentropy,  # 손실 함수 : cross-entropy
    metrics=['acc']
)

model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10, # epochs=10 : 데이터 셋 전체를 10회 반복
    validation_data=(x_test, y_test)
)
# Epoch 1/10
# 17/17 [==============... # 17/17 => batch_size(샘플링 개수, 기본 샘플링 32개)

print(round(model.evaluate(x_test, y_test)[1], 4) * 100, '%')

