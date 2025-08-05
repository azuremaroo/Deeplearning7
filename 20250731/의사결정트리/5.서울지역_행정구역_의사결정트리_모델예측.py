import pandas as pd

district_dict_list = [
    {'district': 'Gangseo-gu', 'latitude': 37.551000, 'longitude': 126.849500, 'label': 'Gangseo'},
    {'district': 'Yangcheon-gu', 'latitude': 37.52424, 'longitude': 126.855396, 'label': 'Gangseo'},
    {'district': 'Guro-gu', 'latitude': 37.4954, 'longitude': 126.8874, 'label': 'Gangseo'},
    {'district': 'Geumcheon-gu', 'latitude': 37.4519, 'longitude': 126.9020, 'label': 'Gangseo'},
    {'district': 'Mapo-gu', 'latitude': 37.560229, 'longitude': 126.908728, 'label': 'Gangseo'},

    {'district': 'Gwanak-gu', 'latitude': 37.487517, 'longitude': 126.915065, 'label': 'Gangnam'},
    {'district': 'Dongjak-gu', 'latitude': 37.5124, 'longitude': 126.9393, 'label': 'Gangnam'},
    {'district': 'Seocho-gu', 'latitude': 37.4837, 'longitude': 127.0324, 'label': 'Gangnam'},
    {'district': 'Gangnam-gu', 'latitude': 37.5172, 'longitude': 127.0473, 'label': 'Gangnam'},
    {'district': 'Songpa-gu', 'latitude': 37.503510, 'longitude': 127.117898, 'label': 'Gangnam'},

    {'district': 'Yongsan-gu', 'latitude': 37.532561, 'longitude': 127.008605, 'label': 'Gangbuk'},
    {'district': 'Jongro-gu', 'latitude': 37.5730, 'longitude': 126.9794, 'label': 'Gangbuk'},
    {'district': 'Seongbuk-gu', 'latitude': 37.603979, 'longitude': 127.056344, 'label': 'Gangbuk'},
    {'district': 'Nowon-gu', 'latitude': 37.6542, 'longitude': 127.0568, 'label': 'Gangbuk'},
    {'district': 'Dobong-gu', 'latitude': 37.6688, 'longitude': 127.0471, 'label': 'Gangbuk'},

    {'district': 'Seongdong-gu', 'latitude': 37.557340, 'longitude': 127.041667, 'label': 'Gangdong'},
    {'district': 'Dongdaemun-gu', 'latitude': 37.575759, 'longitude': 127.025288, 'label': 'Gangdong'},
    {'district': 'Gwangjin-gu', 'latitude': 37.557562, 'longitude': 127.083467, 'label': 'Gangdong'},
    {'district': 'Gangdong-gu', 'latitude': 37.554194, 'longitude': 127.151405, 'label': 'Gangdong'},
    {'district': 'Jungrang-gu', 'latitude': 37.593684, 'longitude': 127.090384, 'label': 'Gangdong'}
]

train_df = pd.DataFrame(district_dict_list)
train_df = train_df[['district','longitude','latitude','label']]
print(train_df)

dong_dict_list = [
    {'dong': 'Gaebong-dong', 'latitude': 37.489853, 'longitude': 126.854547, 'label': 'Gangseo'},
    {'dong': 'Gochuk-dong', 'latitude': 37.501394, 'longitude': 126.859245, 'label': 'Gangseo'},
    {'dong': 'Hwagok-dong', 'latitude': 37.537759, 'longitude': 126.847951, 'label': 'Gangseo'},
    {'dong': 'Banghwa-dong', 'latitude': 37.575817, 'longitude': 126.815719, 'label': 'Gangseo'},
    {'dong': 'Sangam-dong', 'latitude': 37.577039, 'longitude': 126.891620, 'label': 'Gangseo'},

    {'dong': 'Nonhyun-dong', 'latitude': 37.508838, 'longitude': 127.030720, 'label': 'Gangnam'},
    {'dong': 'Daechi-dong', 'latitude': 37.501163, 'longitude': 127.057193, 'label': 'Gangnam'},
    {'dong': 'Seocho-dong', 'latitude': 37.486401, 'longitude': 127.018281, 'label': 'Gangnam'},
    {'dong': 'Bangbae-dong', 'latitude': 37.483279, 'longitude': 126.988194, 'label': 'Gangnam'},
    {'dong': 'Dogok-dong', 'latitude': 37.492896, 'longitude': 127.043159, 'label': 'Gangnam'},

    {'dong': 'Pyoungchang-dong', 'latitude': 37.612129, 'longitude': 126.975724, 'label': 'Gangbuk'},
    {'dong': 'Sungbuk-dong', 'latitude': 37.597916, 'longitude': 126.998067, 'label': 'Gangbuk'},
    {'dong': 'Ssangmoon-dong', 'latitude': 37.648094, 'longitude': 127.030421, 'label': 'Gangbuk'},
    {'dong': 'Ui-dong', 'latitude': 37.648446, 'longitude': 127.011396, 'label': 'Gangbuk'},
    {'dong': 'Samcheong-dong', 'latitude': 37.591109, 'longitude': 126.980488, 'label': 'Gangbuk'},

    {'dong': 'Hwayang-dong', 'latitude': 37.544234, 'longitude': 127.071648, 'label': 'Gangdong'},
    {'dong': 'Gui-dong', 'latitude': 37.543757, 'longitude': 127.086803, 'label': 'Gangdong'},
    {'dong': 'Neung-dong', 'latitude': 37.553102, 'longitude': 127.080248, 'label': 'Gangdong'},
    {'dong': 'Amsa-dong', 'latitude': 37.552370, 'longitude': 127.127124, 'label': 'Gangdong'},
    {'dong': 'Chunho-dong', 'latitude': 37.547436, 'longitude': 127.137382, 'label': 'Gangdong'}
]

test_df = pd.DataFrame(dong_dict_list)
test_df = test_df[['dong', 'longitude', 'latitude', 'label']]
print(test_df)

print(train_df['label'].value_counts())
print(test_df['label'].value_counts())

# train_df 의 longitude(경도), latitude(위도)  ==> 학습 데이터
# train_df 의 label  ==> 학습 데이터의 목표(라벨)

# test_df 의 longitude(경도), latitude(위도)  ==> 테스트 데이터
# test_df 의 label  ==> 테스트 데이터의 목표(라벨)

train_df.drop(['district'], axis=1, inplace=True)
test_df.drop(['dong'], axis=1, inplace=True)


X_train = train_df[['longitude','latitude']]
Y_train = train_df[['label']]

# 의사결정트리는 각 특징을 독립적으로 사용하기 떄문에 별다른 전처리 과정 필요 없음

X_test = test_df[['longitude','latitude']]
Y_test = test_df[['label']]

# 사이킷런 의사결정 트리 모델 학습
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


le = preprocessing.LabelEncoder()
Y_encoded = le.fit_transform(Y_train.values.ravel())
print('Y_encoded :', Y_encoded)
print(Y_train.values.ravel())
print(le.classes_)

# 과대적합 회피 파라미터 설정
clf = tree.DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=2,
    min_samples_leaf=2,
    random_state=70).fit(X_train,Y_encoded)

from sklearn.metrics import accuracy_score
pred = clf.predict(X_test)
print('=========pred==========')
print(pred)
print(le.classes_[pred])  # numpy Fancy Indexing 활용
# 예측정확도
print('accuracy : ' + str( accuracy_score(Y_test.values.ravel(), le.classes_[pred]) ) )
#score : 내부적으로 예측 진행 후 평가 결과 반환 , predict() --> accuracy_score()
# Y_test_encoded = le.transform(Y_test.values.ravel())
# print('score  :' , clf.score(X_test, Y_test_encoded ) )

# 임의 데이터 예측 및 scatter 출력
tempxy = [[127.09, 37.66]]
temppred = clf.predict( tempxy )
print(temppred)
print(le.classes_[temppred])

def display_decison_surface(clf, X,  tempxy):
    x_min = X['longitude'].min() - 0.01
    x_max = X['longitude'].max() + 0.01
    y_min = X['latitude'].min() - 0.01
    y_max = X['latitude'].max() + 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.001),
                         np.arange(y_min, y_max, 0.001))
    #print(xx)  # [[126.8395 126.8405 126.8415 ... 127.1585 127.1595 127.1605]
    #print(xx.shape)  #(237, 322)
    np.set_printoptions(threshold=np.inf)
    #print(xx.ravel())
    #print(yy.ravel())
    Z = clf.predict(np.column_stack([xx.ravel(), yy.ravel()]))
    print(Z.shape)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10,7))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    # color 참조 : https://matplotlib.org/stable/gallery/color/named_colors.html
    plt.scatter(tempxy[0][0],tempxy[0][1], c='indigo', edgecolors='black', s=150)

    plt.title('Decision Surface of predict data', fontsize=16)
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.show()

display_decison_surface(clf, X_train, tempxy)