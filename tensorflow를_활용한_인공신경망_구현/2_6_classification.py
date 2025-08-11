# 2_6_classification.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import to_categorical
iris = load_iris()
print(iris.target_names)
x = iris.data
y = iris.target.reshape(-1, 1)
# 퀴즈: y값의 모양을 'target_names': array(['setosa', 'versicolor', 'virginica']
#  이런 형태로 변경해보세요
y = iris.target_names[y]
# 퀴즈 정답에서 문자열로 바뀐것이 원래 문서였다고 가정하고 라벨로 변경
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
y = enc.fit_transform(y)
print(y)