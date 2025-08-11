import keras.losses
import numpy as np
import tensorflow.keras as keras

def show_softmax():
    def softmax_1(z):
        s = np.sum(z)
        return z / s

    def softmax_2(z):
        z = np.e ** z # 음수가 포함된 경우를 위해 오일러 상수의 지수로 변환
        return softmax_1(z)

    # z = np.array([2.0, 1.0, 0.1])
    z = np.array([2.0, -1.0, 0.1]) # 음수일 경우 확률을 낮게 설정
    # print(softmax_1(z))
    print(softmax_2(z))

def softmax_regression_onehot():
    #   시간, 출석
    x = [[1, 2],  # C
         [2, 1],
         [4, 5],  # B
         [5, 4],
         [8, 9],  # A
         [9, 8]]

    #    성적(0:탈락, 1:통과)
    y = [[0, 0, 1],
         [0, 0, 1],
         [0, 1, 0],
         [0, 1, 0],
         [1, 0, 0],
         [1, 0, 0]]

    x = np.array(x)
    y = np.array(y)

    model = keras.Sequential([
        # keras.layers.Dense(3),  # Dense : 행렬 곱셈
        # keras.layers.Activation('softmax')  # Activation('softmax') : 값을 1~0 사이로 변환
        keras.layers.Dense(3, activation='softmax')  # 위 두 작업을 한줄로 표현
    ])

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=.1),
        loss=keras.losses.categorical_crossentropy,  # binary_crossentropy : crossentropy(log 함수를 사용한 손실 계산)
        metrics=['acc']  # acc : 정확도(전체 중 몇개 맞았는지), mae : mean_absolute_error(오차의 제곱, 회귀에서 얼마나 떨어져 있는지)
    )

    model.fit(x, y, epochs=30, verbose=1)

    p = model.predict(x, verbose=0)
    print(p)

    p_arg = np.argmax(p, axis=1)
    y_arg = np.argmax(y, axis=1)
    print(p_arg)
    print(y_arg)
    print('acc : ', np.mean(p_arg == y_arg))

# show_softmax()
softmax_regression_onehot()