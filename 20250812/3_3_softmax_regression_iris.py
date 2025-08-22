import numpy as np
import pandas as pd
import tensorflow.keras as keras
from sklearn import preprocessing, model_selection

# 퀴즈 : iris_onhot.csv 파일에 대해 70% 로 학습하고 30%에 대해 결과를 알려주는 모델을 만드세요
def softmax_regression_onehot():
    def make_xy():
        df = pd.read_csv('iris_onehot.csv')
        # print(df.values)

        return df.values[:, :-3], df.values[:, -3:]

    x, y = make_xy()
    # print(x.shape, y.shape)

    # x = preprocessing.scale(x) # 표준화
    x = preprocessing.minmax_scale(x) # 정규화 사용

    x_train, x_test, y_train, y_test = data = model_selection.train_test_split(x, y, train_size=.7)

    model = keras.Sequential(
        keras.layers.Dense(3, activation='softmax')
    )

    model.compile(
        optimizer=keras.optimizers.SGD(.1),
        loss=keras.losses.categorical_crossentropy,
        metrics=['acc']
    )

    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

def softmax_regression_sparse():
    def make_xy():
        df = pd.read_csv('iris.csv')
        # print(df.values)

        # y = df.values[:, -1:]
        # y[y == "Setosa"] = 0
        # y[y == "Versicolor"] = 1
        # y[y == "Virginica"] = 2

        enc = preprocessing.LabelEncoder()
        y = enc.fit_transform(df.variety)
        print(y.shape)
        print(y[:10])

        # return df.values[:, :-1], np.int32(y)
        return df.values[:, :-1], y

    x, y = make_xy()
    print(x.shape, y.shape)

    # 표준화, 정규화 : 피처들의 스케일이 다를 때 모델의 성능을 향상시키는 데 도움
    # x = preprocessing.scale(x) # 표준화(데이터를 평균이 0, 표준편차가 1인 분포로 변환)
    x = preprocessing.minmax_scale(x) # 정규화 사용(데이터의 범위를 0~1 사이로 변경)

    x_train, x_test, y_train, y_test = data = model_selection.train_test_split(x, y, train_size=.7)

    model = keras.Sequential(
        keras.layers.Dense(3, activation='softmax')
    )

    model.compile(
        optimizer=keras.optimizers.SGD(.01),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=['acc']
    )

    model.fit(
        x_train, y_train,
        epochs=10,
        validation_data=(x_test, y_test) # validation_data : 튜닝 데이터(학습 데이터는 학습용으로 쓰고 학습 후 검진용으로 내가 원하는 결과가 나올때까지 튜닝하는 용도)
    )

    p = model.predict(x_test)
    # print(p)
    p_bool = np.argmax(p, axis=1)
    # print(p_bool)
    # print(y_test.flatten())
    # print(p_bool == y_test)
    print('acc : ', np.mean(p_bool == y_test.flatten()))
    print('acc : ', model.evaluate(x_test, y_test, verbose=0))

# softmax_regression_onehot()
softmax_regression_sparse()
