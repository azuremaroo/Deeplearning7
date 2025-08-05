import numpy as np
import pandas as pd
from sklearn.datasets import load_iris # 데이터셋 로드
from sklearn.model_selection import train_test_split # 데이터셋 분리
from sklearn.svm import SVC # SVM 모델
from sklearn.metrics import  accuracy_score # 모델 정확도 평가

# # === matplotlib 에 한글 폰트 적용하기 시작 ===
# import seaborn as sns
# import platform
import matplotlib.pyplot as plt
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

# 데이터 로드
iris_data = load_iris()

# ===== 데이터 확인 =====
# print(iris_data)
# print(iris_data.keys())
# print(iris_data['feature_names'])
# print(iris_data['data'][:5])
# print(iris_data['target'][:5])
# print(iris_data['target_names']) # [0:'setosa', 1:'versicolor', 2:'virginica']
# ===== ========== =====

iris_df = pd.DataFrame(np.column_stack((iris_data['data'], iris_data['target']) ), # np.column_stack() 두 배열을 하나로 합쳐줌
                       columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid','target'])
# print(iris_df.head())

# 훈련 데이터 추출 : petal_len, petal_wid
train_df = iris_df[['petal_len', 'petal_wid']]
target_df = iris_df[['target']]
# print(train_df[:2], target_df[:2])

# 데이터 분리
train_input, test_input, train_target, test_target = \
    train_test_split(train_df, target_df, test_size=0.3, random_state=42)

# 분류 모델 준비
cost = 1 # 결정 경계를 변경해주는(하드/소프트 마진) 하이퍼 파라미터(설정값: 1)
gamma = 0.5 # 결정 경계를 변경해주는 하이퍼 파라미터(설정값: 0.5)
svc_model = SVC(C=cost, kernel='rbf', gamma=gamma) # 하이퍼 파라미터 설정

# print(train_target.values.ravel()) # .ravel() ==> 2차원 배열을 1차원으로 변경
svc_model.fit(train_input, train_target.values.ravel()) # target 데이터를 1차원 배열로 입력해야 함

# print(svc_model.score(test_input, test_target.values.ravel()))
test_pred = svc_model.predict(test_input)
# print(test_pred) # 모델 예측
# print(test_target.values.ravel()) # 실제 정답
# print(test_pred == test_target.values.ravel())

# 예측
test_pred = svc_model.predict(test_input)
# print( test_pred )
# print('accuracy : ', accuracy_score(test_target.values.ravel(), test_pred) )

# ============================== 시각화 ==============================

lnames = iris_data['target_names']  # 꽃 이름 정보
markers = ['o','^','s']
colors = ['blue','green','red']

#X, Y 좌표(꽃잎 길이, 꽃잎 너비) 학습(train) 데이터 scatter 출력
for i in set(train_target['target']):  # train_target = [0.0,0.0,..,1.0,1.0,..,2.0,2.0]이므로 중복 없이 0.0 , 1.0, 2.0
    idx = np.where(train_target['target'] == i)
    print('idx : ', idx)
    print(idx[0])
    # train_input.iloc[idx[0]] : dataframe
    # train_input.iloc[idx[0]]['petal_len'] : dataframe 의 'petal_len'컬럼 선택 : series
    # 학습데이터 타깃과 일치한 인덱스의 학습데이터 꽃잎 길이(X좌표) 구함
    #print(train_input.iloc[idx[0]]['petal_len'])
    # 학습데이터 타깃과 일치한 인덱스의 학습데이터 꽃잎 너비(Y좌표) 구함
    #print(train_input.iloc[idx[0]]['petal_wid'])
    # iloc => Fancy indexing 으로 train_input.iloc[idx[0]] , train_input.iloc[idx] 둘다 가능
    plt.scatter(train_input.iloc[idx[0]]['petal_len'], train_input.iloc[idx[0]]['petal_wid'],
                c = colors[int(i)], marker= markers[int(i)],
                label = lnames[int(i)]+'(train)', s=80, alpha=0.3)

# # X, Y 좌표(꽃잎 길이, 꽃잎 너비) 테스트(test) 데이터 scatter 출력
for i in set(test_target['target']):  # test_target = [0.0,0.0,..,1.0,1.0,..,2.0,2.0]이므로 중복 없이 0.0 , 1.0, 2.0
    idx = np.where(test_target['target'] == i)
    # 학습데이터 타깃과 일치한 인덱스의 학습데이터 꽃잎 길이(X좌표) 구함
    #print(test_input.iloc[idx]['petal_len'])
    # 학습데이터 타깃과 일치한 인덱스의 학습데이터 꽃잎 너비(Y좌표) 구함
    #print(test_input.iloc[idx]['petal_wid'])

    # X, Y 좌표(꽃잎 길이, 꽃잎 너비) 학습 데이터 scatter 출력
    plt.scatter(test_input.iloc[idx]['petal_len'], test_input.iloc[idx]['petal_wid'],
                marker= markers[int(i)], label = lnames[int(i)]+'(test)',
                s=130, edgecolors='black', facecolors = 'none')
    # edgecolors : 경계선 색, facecolors = 'none' : 속을 비움(색칠 x)

plt.legend(loc='best')
plt.show()