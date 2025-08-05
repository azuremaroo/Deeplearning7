import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier # knn 분류 모델
from sklearn.model_selection import train_test_split # knn 분리 분류

# kaggle 에서 가져온 데이터
# 도미 데이터
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 빙어 데이터
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

length = bream_length + smelt_length # 길이 데이터 병합
weight = bream_weight + smelt_weight # 무게 데이터 병합

fishdata = np.column_stack((length, weight)) # == fish_data = [ [l, w] for l, w in zip(length, weight) ]
# print(fishdata) # 모델 입력 데이터 준비

# print( np.ones((35, ))) # 도미 : 1
# print( np.zeros((14, ))) # 빙어 : 0
fish_target = np.concatenate( [np.ones((35,)), np.zeros((14,))] ) # np.concatenate() : 두 배열 병합
# print(fish_target)

# 학습데이터와 정답데이터를 train / test 데이터로 분할해서 사용
train_input, test_input, train_target, test_target = \
    train_test_split(fishdata, fish_target, random_state=47, stratify=fish_target) # stratify == 셔플 시 타겟 데이터의 비율 유지 옵션

# 특성 데이터의 정규화 ( 표준 점수 정규화 )
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)

train_scaled = (train_input - mean) / std # train 데이터 표준 점수 정규화
print(train_scaled) # 평균이 0이고 표준편차가 1인 데이터
newdata = ([25,150] - mean) / std # 새로운 예측 데이터 정규화
print(newdata)

model = KNeighborsClassifier()
model.fit(train_scaled, train_target) # 정규화된 데이터로 모델 학습
pred = model.predict([newdata])
print(pred)

distance, indexs = model.kneighbors([newdata])

plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(newdata[0], newdata[1], marker='^')
plt.scatter(train_scaled[indexs,0], train_scaled[indexs,1], marker='D') # 임의의 예측 데이터 주변 5개 데이터
plt.show()