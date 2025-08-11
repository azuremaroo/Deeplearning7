import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential, layers, optimizers, losses

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_test.shape, x_train.shape)
# print(x_test[0])

# 정규화 nomalize 0 to 1
x_train = x_train / x_train.max() # x_train.max() == 255, .max() : 배열 내 가장 큰 값을 반환
x_test = x_test / x_test.max()
# print(x_test.max(), x_train.max())

# visualization first x_test image
# plt.imshow(x_test[0], cmap='gray')
# plt.show()

# 정규화 nomalize 0 to 1
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
# print(x_test.max(), x_train.max())

model = Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# to categorical
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
loss, acc = model.evaluate(x_test, y_test)
print(round(acc, 5) * 100, '%')
