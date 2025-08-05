import numpy as np
import pandas as pd

# 소숫점의 과학적 표기법의 사용 억제
np.set_printoptions(suppress=True)

fish_data = pd.read_csv('fish_data.csv') # kaggle fish market data set
# print(fish_data)
# print(fish_data.info())
# print(fish_data['Species'].unique()) # ['Bream' 'Roach' 'Whitefish' 'Parkki' 'Perch' 'Pike' 'Smelt']
print(fish_data[70:80])

# RangeIndex: 159 entries, 0 to 158
# Data columns (total 6 columns):
#  #   Column    Non-Null Count  Dtype
# ---  ------    --------------  -----
#  0   Species   159 non-null    object
#  1   Weight    159 non-null    float64
#  2   Length    159 non-null    float64
#  3   Diagonal  159 non-null    float64
#  4   Height    159 non-null    float64
#  5   Width     159 non-null    float64

# print(fish_data.columns) #Index(['Species', 'Weight', 'Length', 'Diagonal', 'Height', 'Width'], dtype='object')
fish_input = fish_data[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy() # 특성 데이터, DataFrame to numpy array
fish_target = fish_data['Species'].to_numpy() # 타겟 데이터, Series to numpy array, 딥러닝은 수치 데이터만 입력 가능하지만 머신러닝은 문자열도 입력 가능
# print(fish_input)
# print(fish_target) # ==> one-hot encoding 상태로 변경

# 타겟 데이터를 > 수치 데이터 > 원핫인코딩으로 변환 ( 전처리 )
from sklearn.preprocessing import LabelEncoder # 라벨(문자열)을 ==> 수치데이터로 변경
lb = LabelEncoder()
target_enc = lb.fit_transform(fish_target) # 수치 데이터로 변환
# print(target_enc) # [0 0 0 0 0 0 0 0 0 0 0 0 0 ...

from tensorflow.keras.utils import to_categorical
target_onehot = to_categorical(target_enc) # one-hot 인코딩으로 변환
# print(target_onehot) # [[1. 0. 0. 0. 0. 0. 0.] ...
# print(len(target_onehot)) # 159

# 데이터셋 분리
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = \
    train_test_split(fish_input, target_onehot, random_state=42)

# print(train_input.shape) # (119, 5)
# print(test_input.shape) # (40, 5)

print(fish_target)
print(target_onehot)

# 데이터셋 정규화
from sklearn.preprocessing import  StandardScaler
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_input)
test_scaled = scaler.transform(test_input)

# ==============================================================================================

import tensorflow as tf
from tensorflow.keras.models import load_model # 저장된 모델 로드

new_model = load_model('fish_bestmodel.h5') # 설계된 모델과 가중치까지 모두 불러옴
# new_model.summary()

pred_data = np.array([[242.0,    25.4,      30.0,  11.5200,  4.0200], # Bream
                      [200.0,    23.5,      26.8,   7.3968,  4.1272], # Roach
                      [270.0,    26.0,      28.7,   8.3804,  4.2476], # Whitefish
                      [273.0,    25.0,      28.0,  11.0880,  4.1440], # Parkki
                      [5.9,     8.4,       8.8,   2.1120,  1.4080], # Perch
                      ]) # ['Weight', 'Length', 'Diagonal', 'Height', 'Width']
# print(pred_data)

pred_scaled = scaler.transform(pred_data)

for pred_target in new_model.predict(pred_scaled):
    print(lb.classes_[np.argmax(pred_target)])
