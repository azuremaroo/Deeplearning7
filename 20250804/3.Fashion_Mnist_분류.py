from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np

(train_input, train_target), (test_input, test_target) = fashion_mnist.load_data()

# print(len(train_input)) # 60000
# print(len(test_input)) # 10000
# print(test_input.shape) # (10000, 28, 28)
# print(type(test_input)) # <class 'numpy.ndarray'>
# print(test_target.shape) # (10000,)
# print(type(test_target)) # <class 'numpy.ndarray'>

# plt.imshow(train_input[1], cmap='gray')
# plt.show()

# 분류의 종류와 개수 확인 (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000], dtype=int64))
# print(np.unique(train_target, return_counts=True))

# 케라스 합성곱층은 항상 3차원 변환 입력 필요(lows, cols, channels)
# Conv 층 입력을 위한 shape 변경 및 데이터 정규화
train_scaled = train_input.reshape(-1, 28, 28, 1) / 255
test_scaled = test_input.reshape(-1, 28, 28, 1) / 255

# 60000 train dataset 을 train / valdation Dataset 으로 분할
# train dataset ==> 학습용, validation Dataset 모델 검증용 (train 45000 개, validate 15000 개, test 10000 개)
# train 의 1 epoc 학습 완료 이후 validation 데이터로 모델 성능 검증
# 1 epoch 마다 모델 성능을 검증하면서 진행
from sklearn.model_selection import train_test_split
train_scaled, validation_scaled, train_target, validation_target = \
    train_test_split(train_scaled, train_target) # 60000 개를 45000, 15000 로 분할

# print(len(train_scaled)) # 45000
# print(len(validation_scaled)) # 15000
# print(vailedation_scaled.shape) # (15000, 28, 28, 1)

# CNN 모델 설계
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout # FC layer 층에서 필요
from tensorflow.keras.layers import Conv2D, MaxPool2D # Conv Layer 층에서 필요

CnnModel = Sequential()
CnnModel.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', input_shape=(28, 28, 1))) # Conv 층
CnnModel.add(MaxPool2D(pool_size=(2,2))) # 풀링층

CnnModel.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')) # 2번째 Conv 층
CnnModel.add(MaxPool2D(pool_size=(2,2))) # 풀링층

CnnModel.add(Flatten()) # FC layer 를 위해 펼쳐줌
CnnModel.add(Dense(100, activation='relu'))

# 과대적합 방지용으로 Drop-out 층 추가
CnnModel.add(Dropout(0.4)) # 0.4 의 확률로 중간에 신호를 끊음
CnnModel.add(Dense(10, activation='softmax')) # 출력층 : 10 개 카테고리 분류를 위한 10개 뉴런과 softmax 활성화 함수
# CnnModel.summary()

# 모델 컴파일
CnnModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습 중에 최적의 모델을 어떤 경로에 어떤 파일로 저장하고
# epochs 가 진행되더라도 더이상 loss 가 좋아지지 않을 경우 계속 학습하지 말고 조기 종료 시켜줘라는 콜백 기능 제공
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 조기종료 모니터링을 디폴트 val_loss 로 체크
earlystop_cb = EarlyStopping(patience=2, restore_best_weights=True) # loss 가 증가하면 2번까지 기다려보고 그래도 loss 가 계속 증가할 경우 loss 가 가장 적었던 시점으로 복구하라는 옵션
modelcheck_cb = ModelCheckpoint('Fashion_BestModel.h5')

# 학습진행(fit)에 콜백 기능을 넣어서 진행
# 학습진행 중 train acc, train loss, vali acc, val loss 를 내부 메모리(history) 저장
# validation_data=(validation_scaled, validation_target) ==> 모델 성능 검증
history = CnnModel.fit(train_scaled, train_target, epochs=50, verbose=1,
                       validation_data=(validation_scaled, validation_target),
                       callbacks=[earlystop_cb, modelcheck_cb])

print(history.history['loss']) # train 데이터의 loss
print(history.history['val_loss']) # validation 데이터의 loss

import matplotlib.pyplot as plt
plt.plot(history.history['loss'], c='red')
plt.plot(history.history['val_loss'], c='blue')
plt.show()

