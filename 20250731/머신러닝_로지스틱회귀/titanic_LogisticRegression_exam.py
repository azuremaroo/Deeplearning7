import numpy as np
import pandas as pd

pd.set_option('display.max_rows',20)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)


titanic_df = pd.read_csv('titanic_passengers.csv')
print(titanic_df.shape)
print(titanic_df.head())

# 사이킷런 타깃값 문자열 데이터 사용 가능
# Survived 컬럼 데이터를 target 데이터로 활용
# 1 : 생존 survival, 0 : 별세 fail
titanic_df['Survived'] = titanic_df['Survived'].map({1:'survival',0:'fail'})
# 분석에 사용할 특징 데이터 셋 선택
# Sex, Age, Pclass 컬럼 데이터셋이 생존에 영향을 주는걸로 가설

# Sex 컬럼 'female', 'male' 문자열 데이터를  여성 : 1, 남성 : 0 으로 변경
titanic_df['Sex'] = titanic_df['Sex'].map({'female':1, 'male':0})

# 결측치 검사
print(titanic_df.info())
print(titanic_df.loc[titanic_df['Age'].isnull(),['Age']])  # 177개 NAN

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
print(titanic_Info.head(5))
print(titanic_survival.head(5))

# train dataset / test dastaset : 훈련,테스트 데이터셋 분리
from sklearn.model_selection import train_test_split
train_input,test_input,train_target,test_target = \
    train_test_split(titanic_Info, titanic_survival, random_state=42)

# 스케일 변환
# StandardScaler : 모든 값이 평균 0, 표준편차가 1인 정규분포로 변환
# MinMaxScaler : 최소값 0, 최대값 1로 변환
# RobustScaler : 중앙값 과 IQR(interquartile range): 25%~75% 사이의 범위 사용해 변환
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print('='*80)
print(train_input[:5]) # 0번째 열(Sex) 데이터 표준편차 1
train_scaled = scaler.fit_transform(train_input)
# fit() 했음으로 transform만 하면됨
test_scaled = scaler.transform(test_input)
print(train_scaled) # numpy.ndarray 변환 출력
print('='*80)
print(train_scaled[:,0].std()) # 0번째 열(Sex) 데이터 표준편차 1

# 모델 생성 및 평가
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()  # 모델 생성
lr_model.fit(train_scaled, train_target)  # 훈련

# 훈련 셋 평가
print('train score : %.4f' %lr_model.score(train_scaled, train_target)) # 0.7949101796407185
# 테스트 셋 평가
print('test score : %.4f' %lr_model.score(test_scaled, test_target)) # 0.8026905829596412
# 모델 훈련 계수(가중치, 절편)
print(lr_model.coef_, lr_model.intercept_)

# 예측
print(lr_model.predict(train_scaled[:5]))
# 예측 확률
print(lr_model.predict_proba(train_scaled[:5]))
print(lr_model.classes_)

# decisions = lr_model.decision_function(train_scaled[:5])
# print(decisions)
#
# from scipy.special import expit
# print(expit(decisions))
#
# # 임의 데이터 생성 후  생존 예측
#
# sampledf = pd.DataFrame({'Sex':[1,1,0],'Age':[15, 30, 60],'Class_1':[1,0,1],
#                          'Class_2':[0,1,0]}, index=['Kim','Hong','Park'])
# print(sampledf)
#
# # 스케일 변환
# sample_scaled = scaler.transform(sampledf)
# print(sample_scaled)
#
# # 예측
# print(lr_model.predict(sample_scaled))
#
# # 예측 확률 확인
# print(lr_model.predict_proba(sample_scaled))