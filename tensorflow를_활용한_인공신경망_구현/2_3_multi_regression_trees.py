# 2_3_multi_regression_trees.py
f = open("data/trees.csv", encoding="utf-8")
for row in f:
    print(row.strip())

import pandas as pd, numpy as np
trees = pd.read_csv("data/trees.csv", index_col=0)
                    # header=["Girth", "Height", "Volume"])
x = np.transpose([trees.Girth.values, trees.Height.values])
y = np.reshape(trees.Volume.values, [-1, 1])
print("x =\n",x[:5], "\ny =", y[:5] )
import keras
model = keras.Sequential()
model.add(layer=keras.layers.Dense(1))
model.compile(keras.optimizers.SGD(.0001), keras.losses.mean_squared_error)
model.fit(x, y, None, 10)
# 퀴즈 : 둘레가 10이고 높이가 70일때랑 둘레가 20이고 높이가 80일때 볼륨을 구하는 코드를 쓰시오,.
x_test = np.array([[10, 70], [20, 80]])
p=model.predict(x_test)
print(p)