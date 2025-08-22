import numpy as np
import tensorflow.keras as keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28) 28X28 픽셀 이미지가 60000개, 3차원 데이터라 기존 학습 모델에 사용 불가능
# print(y_train.shape, y_test.shape) # (60000,) (10000,)
# print(np.unique(y_train, return_counts=True))

x_train = x_train.reshape(-1, 784) # 784 = 28 * 28
x_test = x_test.reshape(-1, 784)

# 정규화 스케일링
x_train = x_train / 255 # 이미지 비트 최댓값
x_test = x_test / 255

model = keras.Sequential([
    keras.layers.Input(shape=[784]),
    # layer       x            w(feature, class)
    # ()       = (60000, 784) @ ()               # ? 몇개이든 상관없다는 뜻(실제로 들어오는 값은 6만개지만 영향을 미치지 않음)
    # (?, 512) = (?, 784)     @ (784, 512) + 512 # params : 401920 = 784 * 512 + 512
    keras.layers.Dense(512, activation='relu'),  # 기울기 소실(Vanishing Gradient) 문제 때문에 중간 레이어에 softmax 와 같이 값을 작아지게 하는 함수 사용 안함 => 기울기 소실 해결 : relu 활성화 함수 사용
    # (?, 64) = (?, 512) @ (512, 64) + 64        # params : 32832 = 512 * 64 + 64(bias)
    keras.layers.Dense(64, activation='relu'),
    # (?, 10) = (?, 64) @ (64, 10) + 10          # params : 650 = 64 * 10 + 10
    keras.layers.Dense(10, activation='softmax')
])

model.summary()
# exit()

model.compile(
    # optimizer=keras.optimizers.Adam(.001),
    optimizer=keras.optimizers.RMSprop(.001),
    loss=keras.losses.sparse_categorical_crossentropy,
    metrics=['acc']
)

model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_data=(x_test, y_test)
)