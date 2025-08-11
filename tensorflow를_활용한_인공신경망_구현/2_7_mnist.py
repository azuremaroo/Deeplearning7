# 2_7_mnist.py
from keras.datasets import cifar10 as mnist
from keras import Sequential, layers, optimizers, losses
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_test.shape, x_train.shape)
print(x_train[:, :, :, 1].shape)
x_train = x_train[:, :, :, 1]
x_test = x_test[:, :, :, 1]
x_train = x_train / x_train.max()
x_test = x_test / x_test.max()
# visualization first x_test image
import matplotlib.pylab as plt
plt.imshow(x_test[0], cmap='gray')
plt.show()
# normalize 0 to 1
x_train = x_train.reshape(-1, 32*32)
x_test = x_test.reshape(-1, 32*32)
print(x_test.shape, x_train.shape)
model = Sequential([
    layers.Dense(512, 'relu'),
    layers.Dense(256, 'relu'),
    layers.Dense(10, 'softmax'),
])
model.compile(optimizers.AdamW(0.0001), 'categorical_crossentropy', metrics=['acc'])
# to categorical
from keras.utils import to_categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
model.fit(x_train, y_train, epochs=20, verbose=0, validation_split=0.2)
loss, acc = model.evaluate(x_test, y_test)
print(round(acc, 5) * 100, '%')