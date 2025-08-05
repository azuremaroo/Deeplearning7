import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential # 딥러닝 모델 설계
from tensorflow.keras.layers import Dense # 층(layer) 설계
from tensorflow import optimizers

# ================ tensorflow Gpu 사용 메시지 미출력 ================
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = 2
# ==================================================================

X = np.linspace(0, 10, 10)
# print(X.shape, type(X.shape)) # (10,) <class 'tuple'>

# Y = X + np.random.randn(*X.shape)
# print(Y)
Y = np.array([-1.3648294,   0.44048394,  2.51068021,  2.3977807,   4.01287318,  5.0218039,
  7.15496007,  7.50136673,  9.2238727,  10.46819342])

# 모델 설계
mymodel = Sequential()
# # 첫번째 층은 ==> 꼭 input_dim 또는 input_shape 을 명시해야 함
# mymodel.add(Dense(
#     units=30,  # units : 해당 층의 뉴런 개수, 뉴런(가중치의 합 정보를 가지고 있는 역할)
#     activation='relu', # activation : 활성화 함수 명시, 'relu' 은닉층의 활성화 함수
#     input_dim=4)
# ) # 설계된 은닉층을 추가
mymodel.add(Dense(input_dim=1, units=1, use_bias=False))

# mymodel.summary() # 현재 설계된 모델 요약을 출력
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 30)                150
# =================================================================
# Total params: 150 # == 입력 4 * 뉴런 30 + bios 30
# Trainable params: 150
# Non-trainable params: 0
# _________________________________________________________________
#
# mymodel.add(Dense(1, activation='sigmoid')) # 출력층 추가
# mymodel.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 30)                150
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 31
# =================================================================
# Total params: 181 # == 150 + 30 * 1 + 1 학습될 가중치는 총 181개
# Trainable params: 181
# Non-trainable params: 0
# _________________________________________________________________
#
# Process finished with exit code 0

# 모델 컴파일(환경 설정) : 옵티마이저(adam), 손실함수
# mymodel.compile(
#     optimizer='adam', # 학습 최적화를 위한 옵티마이저 설정(adam 으로 통일)
#     loss='binary_crossentropy', # 학습 과정에서의 손실(비용) 함수 설정(mse, binary_crossentropy, categorical_corssentropy)
#     metrics=['accuracy']
# )
sgd = optimizers.SGD(learning_rate=0.01)
mymodel.compile(optimizer=sgd, loss='mse')

# 학습(fit)
# mymodel.fit(
#     train_input,    # 훈련 데이터
#     target_input,   # 레이블(타겟) 데이터
#     batch_size=10,    # 배치 크기
#     epochs=10,        # 전체 데이터의 총 훈련 회수 지정
#     verbose=1       # 훈련 과정 출력 여부
# ) # 모델 훈련

weight = mymodel.layers[0].get_weights()
w = weight[0][0][0]
print('w begin fit() : ', w) # -1.1590405

mymodel.fit(X, Y, batch_size=4, epochs=50, verbose=1)

weight = mymodel.layers[0].get_weights()
w = weight[0][0][0]
print('w begin fit() : ', w) # 1.007898 1.0154423 1.0193307 1.045921

# import matplotlib.pyplot as plt
# plt.plo
# 평가(evaluate)


# 예측(predict)





