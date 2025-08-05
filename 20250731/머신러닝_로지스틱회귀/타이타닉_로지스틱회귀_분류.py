import numpy as np
import pandas as pd

pd.set_option('display.max_rows',20)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)


titanic_df = pd.read_csv('titanic_passengers.csv')
# print(titanic_df.shape)
# print(titanic_df.head())
# print(titanic_df.columns)

# ============== 데이터 전처리 시작 ==============
# sklearn 은 타겟이 문자열(lavel)이어도 분류가 가능
titanic_df['Survived'] = titanic_df['Survived'].map({1:'servived', 0:'fail'})

# 결측치 채우기
titanic_df['Sex'] = titanic_df['Sex'].map({'female':1, 'male':0})

# 결측치 데이터를 특정 값으로 채우기 ==> fillna
# 결측치 버리기 ==> dropna
titanic_df['Age'].fillna(value=titanic_df['Age'].mean(), inplace=True) # 결측치를 평균값으로 채움
# titanic_df.info()

# 3 등석 항목을 뺀 1등석과 2등석 정보만 활용
onehot_pcclass = pd.get_dummies(titanic_df['Pclass'], prefix='Class') # get_dummies() ==> 데이터 종류별 컬럼을 가진 2차원 배열 생성
titanic_df = pd.concat([titanic_df, onehot_pcclass], axis=1)
# print(titanic_df.head())

# 최종 준비된 데이터 셋
train_input = titanic_df[['Sex','Age','Class_1','Class_2']] # 모델 입력 데이터
train_target = titanic_df[['Survived']] # 정답 데이터

print(train_input.head())

# ============== 데이터 전처리 끝 ==============

# train / test 데이터셋 분할
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = \
    train_test_split(train_input, train_target, random_state=42)

# print(train_input.shape)
# print(train_input[:3])
# print(test_input.shape)
# print(test_input[:3])

# 특성 데이터의 정규화
from sklearn.preprocessing import StandardScaler # 표준점수 정규화 라이브러리
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_input) # train 데이터 정규화
test_scaled = scaler.transform(test_input) # fit 한 데이터와 같은 방법으로 test 데이터 정규화

# print(train_scaled)

# 로지스틱회귀 모델 준비
from sklearn.linear_model import LogisticRegression # 로지스틱회귀(2진 분류) 라이브러리
LR_model = LogisticRegression()

# 모델 학습
LR_model.fit(train_scaled, train_target.values.ravel())

# train 데이터 성능 평가
# print(LR_model.score(train_scaled, train_target.values.ravel())) # 0.7949101796407185

# test 데이터 성능 평가
# print(LR_model.score(test_scaled, test_target.values.ravel())) # 0.8026905829596412

# 예측
# print( test_scaled[:3] )
# print( test_target.values.ravel()[:7] )

# print(LR_model.predict(test_scaled[:7]))
# print(LR_model.classes_) # 모델 예측 클래스 분류 확인 변수
# print(LR_model.predict_proba(test_scaled[:7])) # predict_proba() 각 클래스의 소속 확률 반환

# 새로운 데이터 생성
test_input2 = pd.DataFrame({
    'Sex':[1, 0, 0],
    'Age': [85, 22, 1],
    'Class_1': [0, 1, 0],
    'Class_2': [1, 0, 1],
})

# 새로운 데이터 정규화
test_scaled2 = scaler.transform(test_input2) # fit 한 데이터와 같은 방법으로 test 데이터 정규화

# 새로운 데이터 예측
print(LR_model.predict(test_scaled2))
print(LR_model.classes_) # 모델 예측 클래스 분류 확인 변수
print(LR_model.predict_proba(test_scaled2)) # predict_proba() 각 클래스의 소속 확률 반환









