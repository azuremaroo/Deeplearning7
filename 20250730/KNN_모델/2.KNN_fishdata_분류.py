import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier # knn 분류 모델

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

#1. 데이터 셋 준비(데이터 전처리 과정)
# 학습할 데이터 준비
length = bream_length + smelt_length # 길이 데이터 병합
weight = bream_weight + smelt_weight # 무게 데이터 병합

#데이터 셋 확인
# plt.scatter(length, weight) # 산점도 그래프
# plt.show()

# 학습시킬 데이터 준비
fish_data = [ [l,w] for l, w, in zip(length,weight) ]
# print(fish_data)
fish_data = np.array(fish_data)

# 정답 데이터 준비
fish_target = [1]*35 + [0]*14 # 정답 데이터 ( 도미 35개(1), 빙어 14개(0) )
# print(fish_target)

# 2. 모델 준비
model = KNeighborsClassifier() # n_neighbors=5 ( k 개의 최근접 이웃은 5개로 디폴트 설정되어 있음 )

# 3. 모델 학습
model.fit(fish_data, fish_target) # fit() ==> 훈련, x ==> 모델에 입력할 학습 데이터, y ==> 학습 데이터에 대한 정답 데이터

# 4. 모델 성능 평가
acc = model.score(fish_data, fish_target) # score() ==> 입력데이터에 대한 예측과 정답값을 혼동행렬 비교해서 정확도 반환
# print(acc) # 1.0 ==> 100%

# 5. 완전 새로운 데이터를 가지고 모델 예측(추론)
pred = model.predict([[30,600], [10,11]]) # 1 ==> 도미, 0 ==> 빙어
print(pred) # [1 0]

plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.scatter(30, 600, marker='D')
plt.scatter(10, 11, marker='^')
plt.show()








