from sklearn.metrics import confusion_matrix # 혼동 행렬 생성
from sklearn.metrics import classification_report # 성능평가 결과 확인

# 지도 학습 예제(정답이 있음)
data_target = [ 1, 0, 1, 0, 1, 1, 0, 0 ] # 실제 정답 데이터
model_pred =  [ 1, 1, 1, 0, 1, 0, 1, 0 ] # 모델의 예측값

result = confusion_matrix(data_target, model_pred)
print(result)
# tar pred     # 정답 개수
#     0  1
# 0  [[2 2]
# 1  [1 3]]

result = classification_report(data_target, model_pred)
print(result)
#               precision    recall  f1-score   support         # 성능 평가 지표
#
#            0       0.67      0.50      0.57         4
#            1       0.60      0.75      0.67         4
#
#     accuracy                           0.62         8
#    macro avg       0.63      0.62      0.62         8
# weighted avg       0.63      0.62      0.62         8