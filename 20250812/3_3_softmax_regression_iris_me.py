import numpy as np
import pandas as pd
import tensorflow.keras as keras
from sklearn import preprocessing, model_selection

# 퀴즈 : iris_onhot.csv 파일에 대해 70% 로 학습하고 30%에 대해 결과를 알려주는 모델을 만드세요

def make_xy():
    df = pd.read_csv(
        'iris_onehot.csv',
        header=0,
        names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'Setosa', 'Versicolor', 'Virginica']
    )

    # df.info()
    # print(df.head())

    x = [df.sepal_length, df.sepal_width, df.petal_length, df.petal_width]
    y = [df.Setosa, df.Versicolor, df.Virginica]
    x = np.float32(x)
    x = np.transpose(x)
    y = np.int32(y)
    y = np.transpose(y)

    return x, y

x, y = make_xy()
# print(x[:5, :])
# print(y[:5, :])
# print(x.shape, y.shape)

data = model_selection.train_test_split(x, y, train_size=.7)
x_train, x_test, y_train, y_test = data

model = keras.Sequential([
    keras.layers.Dense(3, activation='softmax')  # 위 두 작업을 한줄로 표현
])

model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=.1),
    loss=keras.losses.categorical_crossentropy,  # binary_crossentropy : crossentropy(log 함수를 사용한 손실 계산)
    metrics=['acc']  # acc : 정확도(전체 중 몇개 맞았는지), mae : mean_absolute_error(오차의 제곱, 회귀에서 얼마나 떨어져 있는지)
)

model.fit(x_train, y_train, epochs=30, verbose=1, validation_data=(x_test, y_test))

p = model.predict(x_test, verbose=0)
print(p)

p_arg = np.argmax(p, axis=1)
y_arg = np.argmax(y_test, axis=1)
print(p_arg)
print(y_arg)
print('acc : ', np.mean(p_arg == y_arg))
print('acc : ', model.evaluate(x_test, y_test, verbose=0))

