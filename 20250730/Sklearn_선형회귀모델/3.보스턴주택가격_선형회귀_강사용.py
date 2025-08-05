import numpy as np
import pandas as pd
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)
pd.set_option('max_colwidth', 1000)
bostondf = pd.read_csv('BostonHousing.csv')
# print(bostondf)
# print(bostondf.info())
bostondf.dropna(axis=0, how='any', inplace=True)
print(bostondf)
print(bostondf.info())
print(bostondf.head(5))
# 입력 데이터 준비 ( 주택가격을 결정짓는 특성 데이터 )
housedata_input = bostondf[['crim','zn','nox','indus','age','dis','rad','tax','chas','rm','ptratio','b','lstat']].copy()
housedata_target = bostondf[['medv']] # 주택가격(타겟)

# 데이터셋 분리
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = \
    train_test_split(housedata_input, housedata_target, random_state=43)
# print(train_input.shape)
# print(test_input.shape)
# print(train_input[:5])
# print(train_target[:5])

train_input = train_input.to_numpy()  # 판단스 데이터프레임을 넘파이 배열로 변경
test_input = test_input.to_numpy()
train_target = train_target.to_numpy()
test_target = test_target.to_numpy()

# 추후에는 특성데이터의 정규화 ==> stardscaler()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_input)
test_scaled = scaler.transform(test_input)

# train_target = train_target.ravel()  # 2차원 배열의 내용을 펼쳐서 1차원을 변경해주는 메서드
# test_target = test_target.ravel()

# 소숫점의 과학적 표기법의 사용 억제
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf) #무한으로 출력합니다. (sys.maxsize 크기 만큼 출력
print(train_scaled[:5])

# 모델준비
from sklearn.linear_model import LinearRegression

LR_model = LinearRegression()  # 선형회귀 모델 준비
# 학습
LR_model.fit(train_scaled, train_target)

print(LR_model.coef_ , LR_model.intercept_)

# 성능평가
print(LR_model.score(test_scaled, test_target))  # 60%

print(test_scaled[:3])
print(test_target[:3])
# 임의의 데이터로 주택 가격 예측
print(LR_model.predict( [[ 0.00376568 , -0.48037651 , 2.87969231 , 1.26781876 , 0.50925525, -1.00026733,
  -0.51014694, -0.00692416 , 3.66375176 ,-0.21055616, -1.72494894, -3.14948833,
   0.36953742]]))