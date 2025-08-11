# 2_5_logistic_regression_pima.py
import pandas as pd
pima = pd.read_csv("data/diabetes.csv", skiprows=9, header=None)
print(pima.head())
X = pima.values[:, :-1]
y = pima.values[:, -1:].astype('F')
print(X.shape, y.shape)
from sklearn import preprocessing
X = preprocessing.scale(X)
print(X.min(), X.max())
train_size = int(len(X) * .7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print(X_train.shape, X_test.shape )
import keras, numpy as np
model = keras.Sequential()
model.add(keras.layers.Dense(1, 'sigmoid'))
model.compile()
model.compile(optimizer=keras.optimizers.SGD(.1), loss=keras.losses.binary_crossentropy, 
              metrics=['acc'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=0)
print(round(model.evaluate(X_test, y_test)[1], 4) * 100, '%')
p = model.predict(X_test, verbose=0)
p_bool = (p > .5)
print("정확도 :", round(np.mean(p_bool == y_test),5)*100, '%')