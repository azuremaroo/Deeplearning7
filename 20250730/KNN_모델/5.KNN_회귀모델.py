import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # 데이터셋 분리
from sklearn.neighbors import KNeighborsRegressor # KNN 회귀 모델

perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# plt.scatter(perch_length, perch_weight)
# plt.show()

# 1. 데이터셋 분할 ==> 이유? 모델의 성능 향상 및 정확한 모델의 성능을 평가하기 위해
train_input, test_input, train_target, test_target = \
       train_test_split(perch_length, perch_weight, random_state=42)

# 2. 입력데이터 차원수 변경 ==> 이유? 모델의 학습과 예측에 데이터를 사용하기 위해 2차원으로 변경
# print(train_input.shape) # (42,) 1차원
# print(test_input.shape) # (14,)
# 현재 train_input 과 test_input 의 1차원인 shape 을 2 차원으로 변경
train_input = train_input.reshape(-1, 1) # -1 입력시 1열의 행 개수는 알아서 설정해줘!!
test_input = test_input.reshape(-1, 1)
# print(train_input.shape) # (42, 1) 2차원
# np.expend_dims() 로도 차원 늘리기 가능

# 3. 회귀 모델 준비
KNN_Reg = KNeighborsRegressor(n_neighbors = 3) # n_neighbors = 3 : 과소 적합을 피하기 위해 최근접 이웃 3개 계산
# KNN_Reg = KNeighborsRegressor() # n_neighbors = 5

# 4. 모델 학습
KNN_Reg.fit(train_input, train_target)

# 5. 모델 성능 평가
print(KNN_Reg.score(test_input, test_target))

# 6. train 데이터의 모델 성능 평가
print(KNN_Reg.score(train_input, train_target))
# n_neighbors = 5(최근접 이웃 5개씩 계산) 일 경우 모델 평가가 훈련 데이터보다 정밀도가 높은 과소 적합 발생!!

# 7. 농어의 길이 50일 때 무게 예측
# print( KNN_Reg.predict([[40]])) # [921.66666667]
print( KNN_Reg.predict([[50]])) # [1033.33333333]
# 길이가 아무리 길어져도 K 개의 최근접 평균을 계산해서 예측하기 때문에 더이상 무게가 증가하지 않는 단점을 가지고 있다!!
distance, indexes = KNN_Reg.kneighbors([[50]])
print(indexes)

plt.scatter(train_input, train_target)
plt.scatter(train_input[indexes], train_target[indexes], marker='D')
plt.show()
