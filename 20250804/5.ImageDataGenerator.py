import numpy as np

train_dir = './CNN_DataSet/cnn_cats_and_dogs_dataset/train'

# 이미지 데이터 증강
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 이미지 데이터 증강 요소 설정
train_image_generator = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=90,
    height_shift_range=0.2 # 이미지를 수직방향으로 무작위 이동(height_shift_range=0.2 : 이미지 높이의 ±20% 범위 내에서 무작위로 이동)
)

# 디렉토리 참조해서 이미지 불러오고 증강시킴
train_data_gen = train_image_generator.flow_from_directory(
    train_dir, # 증강시킬 원본 이미지 위치
    batch_size=2, # 해당 디렉토리에서 몇개의 이미지를 불러올지 결정
    shuffle=True, # 순서대로(False) or 섞어서(True) 불러오기
    save_to_dir='./CNN_DataSet/cnn_cats_and_dogs_dataset/temp', # 증강된 이미지를 해당 위치에 저장
    save_prefix='gen', # 증강이미지 파일 앞에 붙는 이름
    save_format='jpg',
    target_size=(150,150) # 모델에 입력될 크기
)

# print(np.unique(train_data_gen.classes))
# 증강된 이미지 확인
# 실제 학습 코드에서는 fit(train_data_gen) 내 포함된 작업
# i = 0
# for b in train_data_gen: # 반복문에 의해 generator 가 동작
#     i += 1
#     if i > 5:
#         break
