import numpy as np
import pandas as pd
import tensorflow.keras as keras
from sklearn import preprocessing, model_selection

np.set_printoptions(suppress=True) # 소숫점의 과학적 표기법의 사용 억제

def make_xy():
    names = [
        'age',
        'workclass',
        'fnlwgt',
        'education',
        'education_num',
        'marital_status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital_gain',
        'capital_loss',
        'hours-per-week',
        'native-country',
        'target'
    ]
    df = pd.read_csv('adult.data', header=None, names=names)
    # print(df)
    # print(df.describe())
    # df.info()

    # LabelEncoder, LabelBinarizer
    scaler = preprocessing.LabelEncoder()
    workclass = scaler.fit_transform(df.workclass)
    education = scaler.fit_transform(df.education)
    x = [df.age, workclass, df.fnlwgt, education, df.education_num]
    x = np.float32(x)
    # print(x.shape)
    x = x.transpose()
    # print(x.shape)
    y = scaler.fit_transform(df.target)
    y = y.reshape(-1, 1)
    return x, y

x, y = make_xy()
print(x.shape, y.shape) # (32561, 3) (32561, 1)
exit()

data = model_selection.train_test_split(x, y, train_size=.7)
x_train, x_test, y_train, y_test = data

model = keras.Sequential()
model.add(keras.layers.Dense(1, activation='sigmoid')) # 행렬 곱셈 후 sigmoid 로 값을 0 또는 1로 반환

model.compile(
    optimizer=keras.optimizers.Adam(.0001),
    loss=keras.losses.binary_crossentropy,
    metrics=['acc']
)

model.fit(
    x_train, y_train,
    # batch_size=32,
    epochs=20,
    validation_data=(x_test, y_test)
)

p = model.predict(x_test[:5, :])
# print(p)
# print(y_test[:5, :])

p_bool = np.int32(p > 0.5)
print('p_bool : ', p_bool)

equals = (y_test[:5, :] == p_bool)
print('equals : ',equals)
print('acc : ', np.mean(equals))
