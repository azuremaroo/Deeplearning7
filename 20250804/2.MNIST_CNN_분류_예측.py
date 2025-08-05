import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
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
# train_input = train_input.reshape(-1, 28,28,1) / 255 # 최대값이 255 임으로 255로 나누어서 정규화
# print(train_input.shape) # (60000, 28, 28, 1)
test_input = test_input.reshape(-1, 28,28,1) / 255
# print(test_input.shape) # (60000, 28, 28, 1)

# ================================================================================================================
import tensorflow as tf
from tensorflow.keras.models import load_model # 저장된 모델 로드

new_model = load_model('MnistBestModel.h5') # 설계된 모델과 가중치까지 모두 불러옴
# new_model.summary()

# 예측할 이미지 미리 확인
plt.imshow(test_input[4], cmap='gray') # cmap='gray' ==> 흑백 이미지 출력 옵션
plt.show()

plt.imshow(test_input[5], cmap='gray') # cmap='gray' ==> 흑백 이미지 출력 옵션
plt.show()

pred = new_model.predict(test_input[4:6]) # 2장 예측
# print(pred)
# print(pred.shape)
print(np.argmax(pred[0])) # 예측한 정답 확인
print(np.argmax(pred[1])) # 예측한 정답 확인

