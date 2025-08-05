import cv2
import numpy as np

np.set_printoptions(threshold=np.inf) #무한으로 출력합니다. (sys.maxsize 크기 만큼 출력

# ================ 테스트할 이미지 확인 ================
img = cv2.imread('sandal_1.jpg', cv2.IMREAD_GRAYSCALE) # jpg 파일을 그레이스케일로 로딩해서 수치화
# print(img)
# print(img.shape) # (330, 330)

# resize() 를 이용해 cv2 이미지 사이즈 변경 ==> 학습한 이미지 크기인 (28, 28) 로 변경
img = cv2.resize(img, dsize=(28,28), interpolation=cv2.INTER_AREA)
# print(img.shape)

# 학습한 이미지는 바탕이 검정색이었기 때문에 색 반전을 위해 비트 반전 ㄱㄱ
new_img = cv2.bitwise_not(img) # 색상 반전

# cv2.imshow('sandal', new_img) # 'sandal' : 출력창에 표시할 라벨, img : 실제 출력할 이미지 데이터
# k = cv2.waitKey(0)
# if k == 27: # esc 키 입력
#     print(k)
#     cv2.cv2.destroyAllWindows()
# ================ ================ ===================

