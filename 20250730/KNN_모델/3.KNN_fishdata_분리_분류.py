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

# 0.25 비율로 분할, 디폴트 suffle 동작 해서 데이터가 도미, 빙어 섞여 있음
# print(len(train_input))
# print(len(test_input))
# print(test_input)
# print(test_target)

# print(test_input)
print(np.mean(train_input, axis=0)) # 평균 계산
print(np.std(train_input, axis=0)) # 표준편차 계산
# 각 특성 - 평균 / 표준편차

# =========== 도미를 빙어로 예측하는 잘못을 하고 있는 모델 ===========
knmodel = KNeighborsClassifier()
knmodel.fit(train_input, train_target) # 학습

print(knmodel.score(test_input, test_target)) # 성능평가
print(knmodel.predict([[25,150]])) # 임의의 데이터로 예측, [0.] 빙어로 예측

# 특성 데이터의 편차가 너무 큰 문제
# 특정 데이터의 편차를 최소화 시켜주는 정규화 필요!!
# 문제점 해결 방안 ==> 데이터 셋의 정규화

# 이웃까지의 거리와 이웃 샘플의 인덱스 반환
# [25,150] 데이터 주변 5개 확인
distance, indexs = knmodel.kneighbors([[25,150]])
print("distance : ", distance)
print("indexs : ", indexs)
# print(train_input[indexs,0]) # 주변 5개의 길이
# print(train_input[indexs,1]) # 주변 5개의 무게

# 주변 데이터 산점도
plt.scatter(train_input[:,0], train_input[:,1]) # train 의 전체 데이터
plt.scatter(25, 150, marker='^') # 임의의 예측 데이터
plt.scatter(train_input[indexs,0], train_input[indexs,1], marker='D') # 임의의 예측 데이터 주변 5개 데이터
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
#  ===========  ===========  ===========  ===========  ===========