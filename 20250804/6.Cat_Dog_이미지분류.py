train_dir = './CNN_DataSet/cnn_cats_and_dogs_dataset/train'
test_dir = './CNN_DataSet/cnn_cats_and_dogs_dataset/test'

# 이미지 데이터 증강
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 이미지 데이터 증강 요소 설정
train_image_generator = ImageDataGenerator(
    rescale=1.0/255,    # 스케일 조정
    rotation_range=90,  # 회전
)

test_image_generator = ImageDataGenerator(
    rescale=1.0/255,
)

# 디렉토리 참조해서 이미지 불러오고 증강시킴
train_data_gen = train_image_generator.flow_from_directory(
    train_dir, # 증강시킬 원본 이미지 위치
    batch_size=20, # 해당 디렉토리에서 몇개의 이미지를 불러올지 결정
    class_mode='binary', # 분류 결과가 개, 고양이 2종
    target_size=(150,150) # 모델에 입력될 크기
)

test_data_gen = test_image_generator.flow_from_directory(
    test_dir, # 증강시킬 원본 이미지 위치
    batch_size=20, # 해당 디렉토리에서 몇개의 이미지를 불러올지 결정
    class_mode='binary', # 개 아니면 고양이 2종
    target_size=(150,150) # 모델에 입력될 크기
)

# Conv 신경망 설계
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, Dropout, Flatten, Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3,3), input_shape=(150,150,3), # input_shape=(150,150,3) 컬러이므로 rgb 값 3 추가
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 저장, 조기 종료 콜백 추가
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint_cb = ModelCheckpoint('best_cats_dogs_classfication_model.h5', save_best_only=True)
early_stopping_cb = EarlyStopping(patience=3, restore_best_weights=True)

history = model.fit(train_data_gen, validation_data=test_data_gen,
                    steps_per_epoch=200, epochs=50, validation_steps=10,
                    verbose=1, callbacks=[checkpoint_cb, early_stopping_cb])

print(history.history['loss']) # train 데이터의 loss
print(history.history['val_loss']) # validation 데이터의 loss