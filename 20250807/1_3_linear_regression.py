import tensorflow.keras as keras
import numpy as np

# 소숫점의 과학적 표기법의 사용 억제
np.set_printoptions(suppress=True)

x = [1, 2, 3]
y = [1, 2, 3]

# 내부 레이어에 개발자가 직접 접근할 일이 없으므로 변수로 만들지 않고 모델 내 직접 선언
# d1 = keras.layer.Dense(1)
# d2 = keras.layer.Dense(1)
# d3 = keras.layer.Dense(1)

# 순차 모델(레이어를 순차대로 접근)
model = keras.Sequential([
    # keras.layer.Dense(1), # 1 : 데이터 1개에 대해 1개를 반환하겠다는 뜻
    # keras.layer.Dense(1),
    keras.layers.Dense(1) # 마지막 레이어의 리턴값 == 예측값
])


model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.1), # SGD(stochastic gradient dexcent), 추가적인 데이터가 있는 경우 클래스로 전달, lr == 학습률
    loss=keras.losses.mean_squared_error, # mean_squared_error : 손실에 제곱하기
)

model.fit(
    x, y,
    epochs=10,
    verbose=1 # 0 : 학습과정 출력 안함, 1 : 전체 출력, 2 : 학습 프로그레스바 출력 안함 (데이터셋이 클수록 프로그레스 바 필요)
)

# 퀴즈 : x 가 5, 7 일때 결과를 예측하세요 (predict)
p = model.predict([5, 7]) # 예측값은 입력값과 같은 형태로 입력해야 함
print(p) # 반복횟수가 같아도 답이 다른 이유 : w 의 초기값이 랜덤으로 결정되므로

# 퀴즈 : 예측 결과에 대해 mse를 구하세요
# mse = ((p[0] - 5) ** 2 + (p[1] - 7) ** 2) / len(p)
# print('mse : ', mse)

# mse 구하는 방법
# hx = w * x[i]
# c += (hx - y[i]) ** 2 
# c / len(x)
p = model.predict(x)
print('mse : ', np.mean( (p - y)**2 ))




