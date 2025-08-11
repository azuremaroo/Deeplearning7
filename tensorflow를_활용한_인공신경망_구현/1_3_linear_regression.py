# 1_3_linear_regression.py
# import tensorflow.keras as keras
import keras
import numpy as np

x = [[1],
     [2],
     [3]]
y = [[1],
     [2],
     [3]]

x = np.array(x)
y = np.array(y)

model = keras.Sequential([
    keras.layers.Dense(1)
])

# stochastic gradient descent
model.compile(optimizer=keras.optimizers.SGD(0.1),
              loss=keras.losses.mean_squared_error)

model.fit(x, y, epochs=10, verbose=2)

# 퀴즈
# x가 5와 7일 때의 결과를 예측하세요 (predict)
p = model.predict(np.array([[5], [7]]), verbose=0)
print(p)

# 퀴즈
# 예측 결과에 대해 mse를 구하세요
p = model.predict(x, verbose=0)
# print(p)

print('mse :', np.mean((p - y) ** 2))



