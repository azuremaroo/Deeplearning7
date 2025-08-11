import tensorflow.keras as keras
import numpy as np

# 소숫점의 과학적 표기법의 사용 억제
np.set_printoptions(suppress=True)

# featrue(컬럼 - 시간, 출석 2개) 의 개수만큼 가중치(w)가 존재한다
# y  =      x1 +      x2
# hx = w1 * x1 + w2 * x2 + b
#       1         1        0

#   시간, 출석
x = [[1, 2],
    [2, 1],
    [4, 5],
    [5, 4],
    [8, 9],
    [9, 8]]

#    성적
y = [[3],
     [3],
     [9],
     [9],
     [17],
     [17]]

x = np.array(x)
y = np.array(y)

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
    optimizer=keras.optimizers.SGD(learning_rate=.005), # SGD(stochastic gradient dexcent), 추가적인 데이터가 있는 경우 클래스로 전달, lr == 학습률
    loss=keras.losses.mean_squared_error # mean_squared_error : 손실에 제곱하기
)

model.fit(
    x, y,
    epochs=10,
    verbose=1 # 0 : 학습과정 출력 안함, 1 : 전체 출력, 2 : 학습 프로그레스바 출력 안함 (데이터셋이 클수록 프로그레스 바 필요)
)

# 퀴즈 : 3시간 공부하고 6번 출석한 학생과
# 7시간 공부하고 2번 출석한 학생의 성적을 구하세요

x1 = [[3, 6],
      [7, 2]]

x1 = np.array(x1)

print(model.predict(x1))

layer = model.get_layer(index=0) # 레이어가 하나일 경우 index = 0 사용
w, b = layer.get_weights()
print('w : ', w)
print('b : ', b)
