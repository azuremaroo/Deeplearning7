import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # 데이터셋 분리
from sklearn.linear_model import LinearRegression # 선형회귀 모델

# # === matplotlib 에 한글 폰트 적용하기 시작 ===
# import seaborn as sns
# import platform
# import matplotlib.pyplot as plt
#
# from matplotlib import font_manager, rc
#
# plt.rcParams['axes.unicode_minus'] = False
#
# if platform.system() == 'Darwin':
# 	rc('font', family='AppleGothic')
# elif platform.system() == 'Windows':
# 	path = "C:/Windows/Fonts/malgun.ttf"
# 	font_name = font_manager.FontProperties(fname=path).get_name()
# 	rc('font',family=font_name)
# else:
# 	print("Unknon system...")
# # === matplotlib 에 한글 폰트 적용하기 끝 ===
# === pnadas 출력 제어 옵션 설정 시작 ===
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)
pd.set_option('max_colwidth', 1000)
# === pnadas 출력 제어 옵션 설정 끝 ===
# === 소숫점의 과학적 표기법의 사용 억제 ===
np.set_printoptions(suppress=True)
# ========================================


bostondf = pd.read_csv('BostonHousing.csv')
# print(bostondf)
# bostondf.info()

# 데이터 전처리
bostondf.dropna(axis=0, how='any', inplace=True)
# print(bostondf)
# bostondf.info()
# print(bostondf.head(5))

# 입력 데이터 준비 ( 주택가격을 결정짓는 특성 데이터 )
# crim(자치시(town) 별 1인당 범죄율), indus(비소매업자의 토지소유율), chas(강변근처여부), rm(방개수), ptratio(학생/교사 비율)
# housedata_input = bostondf[['crim', 'indus', 'chas', 'rm', 'ptratio']].copy()
housedata_input = bostondf.iloc[:, :-1].copy()
print(housedata_input.head(5))
housedata_target = bostondf[['medv']] # medv 주택가격(타겟)
print(housedata_target.head(5))

train_input, test_input, train_target, test_target = \
       train_test_split(housedata_input, housedata_target, random_state=43)

# print(train_input.shape)
# print(test_input.shape)
# print(train_input[:5])
# print(test_input[:5])

#=======================================================================================
# # 정규화를 위해 numpy 배열로 변경
# train_input = train_input.to_numpy() # 판다스 데이터프레임을 넘파이 배열로 변경
# test_input = test_input.to_numpy()
# train_target = train_target.to_numpy()
# test_target = test_target.to_numpy()
#
# # 나중에 특성데이터의 정규화까지 한번에 해주는 함수 사용 ==> stardscaler()
# # print(train_target.ravel()) # .ravel() ==> 2차원 배열의 내용을 펼쳐서 1차원으로 변경해주는 메서드
# # train_target = train_target.ravel()
# # test_target = test_target.ravel()
# # print(test_input[:5])
# # print(test_target[:5])
#
# # 모델 준비
# LR_model = LinearRegression()
# LR_model.fit(train_input, train_target)
# # print(LR_model.coef_, LR_model.intercept_) # 가중치, 절편(편향)
#
# # 성능 평가
# print(LR_model.score(test_input, test_target))
#
# # 임의의 데이터로 주택 가격 예측
# print(LR_model.predict([[0.08, 3.5, 1, 8.5, 20.3]]))
#=======================================================================================

# ===============특성 데이터의 수동 정규화 ( 표준 점수 정규화 )===============
# mean = np.mean(train_input, axis=0)
# std = np.std(train_input, axis=0)
# test_mean = np.mean(test_input, axis=0)
# test_std = np.std(test_input, axis=0)
#
# train_scaled = (train_input - mean) / std # train 데이터 표준 점수 정규화
# test_scaled = (test_input - test_mean) / test_std # train 데이터 표준 점수 정규화
# # print(train_scaled) # 평균이 0이고 표준편차가 1인 데이터
#===========================================================================
# ==========특성 데이터의 StandardScaler를 이용한 정규화 ( 표준 점수 정규화 )==========
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_input)
test_scaled = scaler.transform(test_input)
#===================================================================================

newdata = scaler.transform(np.array([[0.00632, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.0900, 1, 296, 15.3, 396.90, 4.98]])) # 새로운 예측 데이터 정규화

# 모델 준비
LR_model = LinearRegression()
LR_model.fit(train_scaled, train_target)
# print(LR_model.coef_, LR_model.intercept_) # 가중치, 절편(편향)

# 성능 평가
print(LR_model.score(test_scaled, test_target))

# 임의의 데이터로 주택 가격 예측
print(LR_model.predict([newdata]))

# model = KNeighborsClassifier()
# model.fit(train_scaled, train_target) # 정규화된 데이터로 모델 학습
# pred = model.predict([newdata])
# print(pred)
#
# distance, indexs = model.kneighbors([newdata])
#
# plt.scatter(train_scaled[:,0], train_scaled[:,1])
# plt.scatter(newdata[0], newdata[1], marker='^')
# plt.scatter(train_scaled[indexs,0], train_scaled[indexs,1], marker='D') # 임의의 예측 데이터 주변 5개 데이터
# plt.show()










