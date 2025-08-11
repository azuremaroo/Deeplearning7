import numpy as np
import tensorflow.keras as keras

np.set_printoptions(threshold=np.inf) #무한으로 출력합니다. (sys.maxsize 크기 만큼 출력
np.set_printoptions(suppress=True) # 소숫점의 과학적 표기법의 사용 억제

def make_xy():
    f = open('cars.csv', encoding='utf-8')
    f.readline()  # 첫행 제거(skip header)

    x, y = [], []
    for row in f:
        # 중요 문자열 함수 : format(문자열 형식 지정), split(나누기), join(합치기), strip(문자열 양 끝 모든 공백 제거)
        print(row.strip().split(',')) # strip() 공백 제거
        _, speed, dist = row.strip().split(',') # _ : 값을 무시하고 싶을 때 사용, unpacking(다중 치환)

        # keras 를 사용하기 위해 1. 데이터 타입 변환, 2. np.array 변환, 3. 2차원 배열로 변환 필요
        x.append(int(speed))
        y.append(int(dist))

    f.close()
    # del x[0], y[0] # 첫행 제거
    # x.pop(0), y.pop(0) # 첫행 제거

    # x = np.reshape(np.float64(x), [-1, 1]) # -1 : ( col 이 1 일때 행을 ) 알아서 결정해라 ( 50이 입력됨 )
    # y = np.reshape(np.float64(y), [-1, 1])

    return np.reshape(x, [-1, 1]), np.reshape(y, [-1, 1])

def vis(p): # 시각화
    import matplotlib.pyplot as plt
    p = np.reshape(p, [-1])
    plt.plot(x, y, 'b*')
    plt.plot([0, 30], [0, p[1]], 'r')
    plt.plot([0, 30], [p[0], p[1]], 'g')
    plt.show()

x, y = make_xy()
# print(x)
# print(y)
# exit()
# vis(p)

model = keras.Sequential()
model.add(keras.layers.Dense(1))
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.001), # SGD(stochastic gradient dexcent), 추가적인 데이터가 있는 경우 클래스로 전달, lr == 학습률
    loss=keras.losses.mean_squared_error # mean_squared_error : 손실에 제곱하기) # compile : 진짜 컴파일이 아니라 환경설정 정도의 의미, sgd : 확률적인 경사 하강(모집단을 전부 제공하지 않음)
)

model.fit(x, y, epochs=50, verbose=1)

for layer in model.layers:
    print('Layer: ', layer.name)
    print('Weights : ', np.round(layer.get_weights()[0], 5)) # 가중치 w
    print('Bias : ', np.round(layer.get_weights()[1], 5)) # 절편 b

# 속도가 30일때와 50일때의 제동거리를 예측하세요
x_test = np.array([[0], [30], [50]])
p = model.predict(x_test)
print('p : ', p)

