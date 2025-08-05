import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# mnist : 0 ~ 9 손글씨 데이터의 이미지를 수치화 시켜놓은 데이터
(train_input, train_target), (test_input, test_target) = mnist.load_data() # 기본 내장된 흑백이미지
# print(len(train_input)) # 60000 개
# print(len(test_input)) # 10000 개
# return_counts=True ==> 각 타깃 정보의 개수까지 반환하는 옵션
# print(np.unique(train_target, return_counts=True)) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# print(train_input[0])
# print(train_input.shape) # (60000, 28, 28)

# plt.imshow(train_input[0], cmap='gray') # cmap='gray' ==> 흑백 이미지 출력 옵션
# plt.show()

# train/test 를 shape 변경하면서 정규화
# (60000, 28, 28) ==> (60000, 28, 28, 1) 으로 shape 을 변경(3차원으로)
train_input = train_input.reshape(-1, 28,28,1) / 255 # 최대값이 255 임으로 255로 나누어서 정규화
# print(train_input.shape) # (60000, 28, 28, 1)
test_input = test_input.reshape(-1, 28,28,1) / 255
# print(test_input.shape) # (60000, 28, 28, 1)

# 모델 설계
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D # Conv2D : 케라스의 합성곱층
from tensorflow.keras.layers import MaxPool2D

Mnist_model = Sequential()

# filters : 커널필터의 개수
# kernel_size : 커널필터의 크기 ( 3x3, 5x5 )
# strides : 디폴트 1 strides
# apdding : same 패딩 사용 ==> 입력과 출력이 동일하도록
# activation : 활성화 함수 지정 ==> relu
# 합성곱의 첫번째층은 이미지의 입력 shape 을 어떻게 받을지 지정, input_shape 지정
Mnist_model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', input_shape=(28, 28, 1))) # 합성곱층
# output(FM) 의 size ==> (28, 28, 32)
Mnist_model.add(MaxPool2D(pool_size=(2,2))) # 풀링층
# output(FM) 의 size ==> (14, 14, 32)
# Mnist_model.summary()

Mnist_model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')) # 합성곱층
Mnist_model.add(MaxPool2D(pool_size=(2,2))) # 풀링층
# Mnist_model.summary()

# FC layer 추가
Mnist_model.add(Flatten()) # Flatten() : FM 을 펼쳐서 하나의 펼친 층을 추가
Mnist_model.add(Dense(100, activation='relu')) # FC 레이어에 히든충 추가
Mnist_model.add(Dense(10, activation='softmax')) # 마지막 다층 분류를 위한 출력층 추가
# Mnist_model.summary()

# 모델 컴파일
# loss='catecorical_crossentropy' ==> 타깃이 원한 인코딩 상태일때 사용
# loss='sparse_categorical_crossentropy' ==> 타깃이 정수 상태일때 사용하는 loss function
Mnist_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습
Mnist_model.fit(train_input, train_target, batch_size=16, epochs=20, verbose=1) # 60000 / 16 = 3750 iter

# 모델 평가
score = Mnist_model.evaluate(test_input, test_target)
print('모델 성능 평가 : ', score[1]) # 정확도 체크

Mnist_model.save('MnistBestModel.h5')
