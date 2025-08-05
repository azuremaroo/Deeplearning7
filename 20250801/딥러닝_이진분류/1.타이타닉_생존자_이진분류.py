import numpy as np
import pandas as pd

pd.set_option('display.max_rows',20)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)

# ============================ 데이터 셋 전처리 ============================
titanic_df = pd.read_csv('titanic_passengers.csv')
# print(titanic_df.shape)
# print(titanic_df.head())

# 결측치 검사
print(titanic_df.info())
print(titanic_df.loc[titanic_df['Age'].isnull(),['Age']])  # 177개 NAN

# Sex 컬럼 'female', 'male' 문자열 데이터를  여성 : 1, 남성 : 0 으로 변경
titanic_df['Sex'] = titanic_df['Sex'].map({'female':1, 'male':0})

# 결측치 채우기 : 평균 값
titanic_df['Age'].fillna(value=titanic_df['Age'].mean(), inplace=True)
print(titanic_df.head(10))

# PClass ==> 1등석, 2등석, 3등석 구분
print('='*80)
onehot_Pclass = pd.get_dummies(titanic_df['Pclass'], prefix='Class')
print(onehot_Pclass)  # Class_1  Class_2  Class_3 컬럼 생성
print(type(onehot_Pclass))
# titanic_df 와 onehot_Pclass 와 결합
titanic_df = pd.concat([titanic_df,onehot_Pclass], axis=1)
print(titanic_df.head(10))

# 데이터셋 준비
# 입력 데이터 셋 : [Sex, Age, Class_1 ,  Class_2 ] 컬럼
# 타깃 데이터 셋 : [Survived] 컬럼

titanic_Info = titanic_df[['Sex','Age','Class_1','Class_2']]
titanic_survival = titanic_df['Survived']
print(titanic_Info.head(5)) # 입력 데이터 사용
print(titanic_survival.head(5)) # 정답
# ====================================================================

# # train dataset / test dastaset : 훈련,테스트 데이터셋 분리
from sklearn.model_selection import train_test_split
train_input,test_input,train_target,test_target = \
    train_test_split(titanic_Info, titanic_survival, random_state=42)

# # 스케일 변환
# StandardScaler : 모든 값이 평균 0, 표준편차가 1인 정규분포로 변환
# MinMaxScaler : 최소값 0, 최대값 1로 변환
# RobustScaler : 중앙값 과 IQR(interquartile range): 25%~75% 사이의 범위 사용해 변환
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# print('='*80)
# print(train_input[:5]) # 0번째 열(Sex) 데이터 표준편차 1
train_scaled = scaler.fit_transform(train_input)
# # fit() 했음으로 transform만 하면됨
test_scaled = scaler.transform(test_input)
# print(train_scaled) # numpy.ndarray 변환 출력
# print('='*80)
# print(train_scaled[:,0].std()) # 0번째 열(Sex) 데이터 표준편차 1
# print(train_scaled[:5])
# print(test_scaled[:5])

# 딥러닝 이진분류 모델 설계
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(units=8, input_dim=4, activation='relu')) # 첫 층
model.add(Dense(units=4, activation='relu')) # 두번째 층
model.add(Dense(units=1, activation='sigmoid')) # 출력층
# model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # 모델 컴파일

model.fit(train_scaled, train_target, batch_size=64, epochs=1000, verbose=1) # 학습

score = model.evaluate(test_scaled, test_target)
print('성능평가 : ', score[1]) # 정확도

model.save('titanic_model.h5') # 학습된 모델 저장

# test_input2 = pd.DataFrame({
#     'Sex':[1, 0, 0],
#     'Age': [85, 22, 1],
#     'Class_1': [0, 1, 0],
#     'Class_2': [1, 0, 1],
# })
#
# # 새로운 데이터 정규화
# test_scaled2 = scaler.transform(test_input2) # fit 한 데이터와 같은 방법으로 test 데이터 정규화
# print(model.predict(test_scaled2))