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



