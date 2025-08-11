# 2_1_linear_regression.py
import numpy as np
import keras, matplotlib.pyplot as plt

# format, split, join, strip
def make_xy():
    f = open('data/cars.csv', encoding='utf-8')

    # skip header
    f.readline()

    x, y = [], []
    for row in f:
        # print(row.strip().split(','))

        _, speed, dist = row.strip().split(',')
        x.append(int(speed))
        y.append(int(dist))

    f.close()
    return np.reshape(x, (-1, 1)), np.reshape(y, (-1, 1))

def vis(p):
    p = np.reshape(p, [-1])
    plt.plot(x, y, 'b*')
    plt.plot([0, 30], [0, p[1]], 'r')
    plt.plot([0, 30], [p[0], p[1]], 'g')
    plt.show()

x, y = make_xy()

model = keras.Sequential()
model.add(keras.layers.Dense(1))

model.compile(optimizer=keras.optimizers.SGD(0.001),
              loss=keras.losses.mean_squared_error)

model.fit(x, y, epochs=10, verbose=2)

# # 속도가 30일때와 50일때의 제동거리를 예측하세요.
# x_test = np.array([[0], [30], [50]])
# vis(model.predict(x_test, verbose=0))
# np.set_printoptions(suppress=True)
# for layer in model.layers:
#     print(f"Layer: {layer.name}")
#     print("Weights:", np.round(layer.get_weights()[0], 5),
#           "Bias:", np.round(layer.get_weights()[1], 5) )