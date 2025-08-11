from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

iris = load_iris()
print(iris)
x = iris.data
y = iris.target.reshape(-1, 1)

# 퀴즈 : y 값의 모양을 [['setosa'], ['versicolor'], ['virginica']] 형태로 변경하기
y = iris.target_names[y]
# print(y)

# 퀴즈 : 정답에서 문자열로 바뀐것이 원래 문서였다고 가정하고 라벨로 변경
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
y = enc.fit_transform(y)
print(y)