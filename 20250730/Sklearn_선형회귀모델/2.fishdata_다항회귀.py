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

train_poly = np.column_stack( (train_input**2, train_input) )
# print(train_poly)
test_poly = np.column_stack( (test_input**2, test_input) )
# print(test_poly.shape) # (14, 2)

lr_model = LinearRegression()

lr_model.fit(train_poly, train_target)

print(lr_model.score(test_poly, test_target)) # 0.9775935108325121

print(lr_model.coef_) # 가중치 확인(w1, w2)
print(lr_model.intercept_) # 편향 확인(b)

print(lr_model.predict([[50**2, 50]])) # [1573.98423528]
print(lr_model.predict([[2**2, 2]])) # [76.99168926]

# 다항회귀의 회귀선 공식 = w1x^2 + w2x + b
# print((lr_model.coef_[0] * 50**2) + lr_model.coef_[1] * 50 + lr_model.intercept_) # 1573.9842352827404

w2 = 1.0 # 가중치 w1
w1 = -21.5 # 가중치 w2
bias = 116 # 편향 b
xpoint = np.arange(15, 51)

# print(xpoint)
# print(w2*(xpoint**2) + w1*xpoint + bias)

plt.scatter(train_input, train_target)
plt.plot(xpoint, w2*(xpoint**2) + w1*xpoint + bias, c='red')
# 길이가 43인 농어의 무게를 계산해서 ^ 마커로 출력
plt.scatter(43, (w2*(43**2) + w1*(43) + bias), marker='^')
plt.show()

