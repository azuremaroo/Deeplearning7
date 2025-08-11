from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

iris = load_iris()
# print(iris)
x = iris.data
y = iris.target.reshape(-1, 1)

# 퀴즈 : y 값의 모양을 [[1,0,0], [0,1,0], ... [0,0,1]] 형태로 변경하기
from tensorflow.keras.utils import to_categorical
y = to_categorical(y, 3) # one-hot 인코딩으로 변환
print(y)

# 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.7, stratify=y)

print(y_test.shape)

import tensorflow.keras as keras
model = keras.Sequential()
model.add(keras.layers.Dense(
    3,  # 출력 노드는 3개(꽃의 종류 수)
    activation='softmax') # softmax : 출력을 확률 분포로 변환(예 : [0.2, 0.7, 0.1])(세 클래스에 대한 확률 합이 1)
)
model.compile(
    optimizer=keras.optimizers.SGD(.088), # 학습률이 0.1인 확률적 경사 하강법 옵티마이저를 사용
    loss=keras.losses.categorical_crossentropy, # 다중 분류 문제에서 정수 형태(0, 1, 2)의 레이블(정답 클래스)을 다루는 손실 함수(one-hot encoding 을 수동으로 한 경우)
    # loss=keras.losses.sparse_categorical_crossentropy, # sparse : 내부적으로 one-hot encoding으로 변환 후 모델의 예측값(softmax의 출력값)과 비교하여 손실 계산
    metrics=['acc'] # 모델의 성능을 정확도로 평가
)
model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))

# test 데이터의 정확도 출력
print('evaluate : ', round(model.evaluate(x_test, y_test)[1], 4) * 100, '%')
p = model.predict(x_test)
print('p : ', p)
# print(round(np.mean((y_test == p.argmax(1).reshape(-1, 1))), 6) * 100, '%')
print('p : ', round(np.mean((y_test.argmax(1) == p.argmax(1))), 6) * 100, '%')
