import numpy as np
import pandas as pd

# 소숫점의 과학적 표기법의 사용 억제
np.set_printoptions(precision=8, suppress=True)

fish_data = pd.read_csv('fish_data.csv') # kaggle fish market data set
# print(fish_data)
# print(fish_data.info())
# print(fish_data['Species'].unique()) # ['Bream' 'Roach' 'Whitefish' 'Parkki' 'Perch' 'Pike' 'Smelt']

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
# print(fish_target)

# 데이터셋 분리
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = \
    train_test_split(fish_input, fish_target, random_state=42)

# print(train_input.shape) # (119, 5)
# print(test_input.shape) # (40, 5)

# 데이터셋 정규화
from sklearn.preprocessing import  StandardScaler
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_input)
test_scaled = scaler.transform(test_input)
# print(train_scaled[:3])
# print(test_scaled[:3])

# 훈련 모델 생성
from sklearn.linear_model import LogisticRegression
lrmodel = LogisticRegression(multi_class='multinomial', # multi_class='multinomial' ==> 소프트맥스(다중 분류 비용 함수) 선택
                             C=20, # 규제 강도를 조절하는 파라미터로, 값이 클수록 규제가 약해집니다. C=20은 상대적으로 높은 값으로, 모델이 훈련 데이터에 더 잘 맞도록(과적합 위험이 있음) 허용합니다.
                             max_iter=1000) # 최적화 알고리즘의 최대 반복 횟수, 기본값보다 높은 1000회로 설정하여, 복잡한 데이터셋에서도 충분히 수렴할 수 있도록

# 모델 학습
lrmodel.fit(train_scaled, train_target)
print(lrmodel.score(train_scaled, train_target)) # 0.9327731092436975
print(lrmodel.score(test_scaled, test_target)) # 0.925
# ==> 과소/과대적합이 없는 딱 좋은 상태

# 모델 예측
print('정답')
print(test_target[5:11])
print('예측')
print(lrmodel.predict(test_scaled[5:11]))
print('예측값 종류')
print(lrmodel.classes_)

# 임의의 물고기 데이터를 예측해서 물고기 종류를 분류해주세요!!
# print(test_input[5:11])
pred_data = np.array([[1., 1., 1., 10., 5.]]) # ['Weight', 'Length', 'Diagonal', 'Height', 'Width']
print(pred_data)
pred_scaled = scaler.transform(pred_data)
print(lrmodel.predict(pred_scaled))


