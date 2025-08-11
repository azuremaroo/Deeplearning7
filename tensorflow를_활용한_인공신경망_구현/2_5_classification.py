# 2_4_classification.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import to_categorical
iris = load_iris()
x = iris.data
y = iris.target.reshape(-1, 1)
print(y)
# 퀴즈: y값의 모양을 [[1, 0, 0], ... [0, 1, 0], ... [0, 0, 1]] 이런 형태로 변경해보세요
y = to_categorical(y, 3)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.7, stratify=y)
print(y_test.shape)
import keras
model = keras.Sequential()
model.add(keras.layers.Dense(3, 'softmax'))
model.compile(optimizer=keras.optimizers.SGD(.088), 
              loss=keras.losses.categorical_crossentropy, metrics=['acc'])
model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), verbose=0)
# test데이터의 정확도를 출력해보세요
print(round(model.evaluate(x_test, y_test)[1], 5) * 100, '%') 
p = model.predict(x_test, verbose=0)
# print(p.argmax(1))
# print("y = \n", y_test)
print(round(np.mean((y_test.argmax(1) == p.argmax(1))), 6)* 100, '%')