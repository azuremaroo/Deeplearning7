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

    #    성적(A[1, 0, 0], B[0, 1, 0], C: [0, 0, 1])
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
        loss=keras.losses.categorical_crossentropy,  # categorical_crossentropy : crossentropy(log 함수를 사용한 손실 계산)
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
    print('acc : ', model.evaluate(x, y, verbose=0))

def softmax_regression_spars():
    #   시간, 출석
    x = [[1, 2],  # C
         [2, 1],
         [4, 5],  # B
         [5, 4],
         [8, 9],  # A
         [9, 8]]

    #    성적(A[1, 0, 0], B[0, 1, 0], C: [0, 0, 1])
    # y = [[0, 0, 1],
    #      [0, 0, 1],
    #      [0, 1, 0],
    #      [0, 1, 0],
    #      [1, 0, 0],
    #      [1, 0, 0]]
    # y = np.argmax(y, axis=1)
    y = [2, 2, 1, 1, 0, 0] # 원핫 백터에 argmax 적용한 값의 배열(타겟이 많아서 식별이 힘들 수 있으므로)

    x = np.array(x)
    y = np.array(y)

    model = keras.Sequential([
        keras.layers.Dense(3, activation='softmax')  # 실제 정답의 종류 개수만큼 3 입력
    ])

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=.1),
        loss=keras.losses.sparse_categorical_crossentropy, # 정답이 원핫벡터로 연산되지만 입력값은 다르므로 알아서 처리하라는 함수
        metrics=['acc']  # acc : 정확도(전체 중 몇개 맞았는지), mae : mean_absolute_error(오차의 제곱, 회귀에서 얼마나 떨어져 있는지)
    )

    model.fit(x, y, epochs=30, verbose=1)

    p = model.predict(x, verbose=0)
    print(p)

    p_arg = np.argmax(p, axis=1)
    print(p_arg)
    print(y)
    print('acc : ', np.mean(p_arg == y))
    print('acc : ', model.evaluate(x, y, verbose=0))


# show_softmax()

softmax_regression_onehot()
softmax_regression_spars()