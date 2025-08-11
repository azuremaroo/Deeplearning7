import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras

# 전치 행렬 : 행과 열의 위치를 바꾸는 것
# (3, 7) => (7, 3)

# print(np.e) # 오일러 상수 : 2.718281828459045

def show_sigmoid():
    def sigmoid(z):
        return 1 / (1 + np.e ** -z)

    # sigmoid 그래프
    for z in np.linspace(-10, 10, 10):
        s = sigmoid(z) # 마지막 레이어의 값을 sigmoid 함수로 필터링하는 과정 의미

        plt.plot(z, s, 'ro')
    plt.show()

# 손실 : 딥러닝이 학습이 잘 되고 있는지 판단하는 기준(손실이 클수록 학습이 잘 안되고 있는 것)

def show_logistic(y): # y : 예측값, sigmoid 에 의해 1 or 0 만 입력됨
    def log_a():
        return 'a'

    def log_b():
        return 'b'

    # if y == 1:
    #     print(log_a())
    # else:
    #     print(log_b())
    print(y * log_a() + (1 - y) * log_b()) # if 대신 수식으로 해결

# 어떤 학습 모델을 사용할지는 y 값에 의해 결정된다(x는 상관없음)
def logistic_regression():
    #   시간, 출석
    x = [[1, 2],    # 탈락
         [2, 1],
         [4, 5],    # 통과
         [5, 4],
         [8, 9],
         [9, 8]]

    #    성적(0:탈락, 1:통과)
    y = [[0],
         [0],
         [1],
         [1],
         [1],
         [1]]

    x = np.array(x)
    y = np.array(y)

    model = keras.Sequential([
        # keras.layers.Dense(1),  # Dense : 행렬 곱셈
        # keras.layers.Activation('sigmoid')  # Activation('sigmoid') : 값을 1, 아니면 0 으로 변환
        keras.layers.Dense(1, activation='sigmoid') # 위 두 작업을 한줄로 표현
    ])

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=.1),
        loss=keras.losses.binary_crossentropy,  # binary_crossentropy : crossentropy(log 함수를 사용한 손실 계산)
        metrics=['acc'] # acc : 정확도(전체 중 몇개 맞았는지), mae : mean_absolute_error(오차의 제곱, 회귀에서 얼마나 떨어져 있는지)
    )

    model.fit(x, y, epochs=10, verbose=1)

    # ===== 정확도(acc) 계산 =====
    p = model.predict(x)
    print(p)

    p_bool = np.int32(p > 0.5)
    print(p_bool)

    equals = (y == p_bool)
    print(equals)
    print('acc : ', np.mean(equals))
    # ============================

# show_sigmoid()
# show_logistic(y=1) # sigmoid 에 의해
# show_logistic(y=0)
logistic_regression()
