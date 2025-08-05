import tensorflow as tf
from tensorflow.keras.models import load_model # 저장된 모델 로드
import numpy as np

np.set_printoptions(threshold=np.inf) #무한으로 출력합니다. (sys.maxsize 크기 만큼 출력

bestcnn_model = load_model('best_cats_dogs_classfication_model.h5') # 설계된 모델과 가중치까지 모두 불러옴

from tensorflow.keras.preprocessing import image

# 임의의 예측 데이터 로딩
# test_dog.jpg 인터넷에서 강아지 이미지 다운받아서 해당 위치에 저장
img = image.load_img('./CNN_DataSet/cnn_cats_and_dogs_dataset/predict_img/t1.jpg', target_size=(150,150))
img_arr = image.img_to_array(img) / 255.0 # 데이터 정규화
# print(img_arr)
# print(img_arr.shape)

# 새로운 이미지 예측
pred = bestcnn_model.predict(img_arr.reshape(1,150,150,3), batch_size=1)
print(pred)

pre_result = np.where(pred[0] > 0.5, 1, 0) # cat label : 0, dog label : 1
cat_dog_classnames = np.array(['cat', 'dog'])

print(cat_dog_classnames[pre_result])