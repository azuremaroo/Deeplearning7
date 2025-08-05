import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # 데이터셋 분리
from sklearn.linear_model import LinearRegression # 선형회귀 모델

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

# 선형회귀 모델 준비
lr_model = LinearRegression()

# 선형회귀 모델 학습
lr_model.fit(train_input, train_target)

# 모델 성능 평가
print(lr_model.score(test_input, test_target))

# 농어의 길이 50일 때 무게 예측
print(lr_model.predict([[50]])) # [1241.83860323]

# 선형회귀 모델이 학습이 완료됐으면
# y = wx + b 의 회귀선에 대한 가중치(w), 절편(b) 값 확인 가능
print(lr_model.coef_) # 39.01714496
print(lr_model.intercept_) # 절편 -709.0186449535474
print(lr_model.coef_ * 50 + lr_model.intercept_) # 회귀선 공식 ==> wx + b 적용

plt.scatter(train_input, train_target) # 산점도
plt.plot([15,50],[lr_model.coef_ * 15 + lr_model.intercept_, lr_model.coef_ * 50 + lr_model.intercept_], c='red')
plt.scatter(90, lr_model.coef_ * 90 + lr_model.intercept_, marker='^')
plt.show()





