import cv2
import numpy as np

np.set_printoptions(threshold=np.inf) #무한으로 출력합니다. (sys.maxsize 크기 만큼 출력

img = cv2.imread('sandal_1.jpg', cv2.IMREAD_GRAYSCALE) # jpg 파일을 그레이스케일로 로딩해서 수치화
# print(img)
# print(img.shape) # (330, 330)

# resize() 를 이용해 cv2 이미지 사이즈 변경 ==> 학습한 이미지 크기인 (28, 28) 로 변경
img = cv2.resize(img, dsize=(28,28), interpolation=cv2.INTER_AREA)
# print(img.shape)

# 비트 반전 : 학습한 이미지는 바탕이 검정색이었기 때문에 색 반전을 위해
new_img = cv2.bitwise_not(img) # 색상 반전

new_img_reshape = new_img.reshape(1,28,28,1) / 255
# print(new_img_reshape.shape) # (1, 28, 28, 1)

from tensorflow.keras.models import load_model

fashion_model = load_model('Fashion_BestModel.h5')

pred = fashion_model.predict(new_img_reshape)

classes = ['티셔츠/탑','바지','스웨터','드레스','코트','샌달','셔츠','스니커즈','가방','앵클부츠']
print(classes[np.argmax(pred)]) # 5 (샌달)

# ========= 다른 이미지 예측 추가 =========
img2 = cv2.imread('onepeice.jpg', cv2.IMREAD_GRAYSCALE) # jpg 파일을 그레이스케일로 로딩해서 수치화
img2 = cv2.resize(img2, dsize=(28,28), interpolation=cv2.INTER_AREA)
new_img2 = cv2.bitwise_not(img2) # 색상 반전
new_img_reshape2 = new_img2.reshape(1,28,28,1) / 255

pred2 = fashion_model.predict(new_img_reshape2)
print('원피스 : ', classes[np.argmax(pred2)]) # 드레스


img3 = cv2.imread('bag1.png', cv2.IMREAD_GRAYSCALE) # jpg 파일을 그레이스케일로 로딩해서 수치화
img3 = cv2.resize(img3, dsize=(28,28), interpolation=cv2.INTER_AREA)
new_img3 = cv2.bitwise_not(img3) # 색상 반전
new_img_reshape3 = new_img3.reshape(1,28,28,1) / 255

pred3 = fashion_model.predict(new_img_reshape3)
print('가방 : ', classes[np.argmax(pred3)]) # 드레스


img4 = cv2.imread('ankleboots2.jpg', cv2.IMREAD_GRAYSCALE) # jpg 파일을 그레이스케일로 로딩해서 수치화
img4 = cv2.resize(img4, dsize=(28,28), interpolation=cv2.INTER_AREA)
new_img4 = cv2.bitwise_not(img4) # 색상 반전
new_img_reshape4 = new_img4.reshape(1,28,28,1) / 255

pred4 = fashion_model.predict(new_img_reshape4)
print('앵클부츠 : ', classes[np.argmax(pred4)])
