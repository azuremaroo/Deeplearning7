# keras 로 kaggle competition 참여하기
import pandas as pd
from sklearn import preprocessing, model_selection
import tensorflow.keras as keras

# leaf 폴더의 train.csv 파일을 읽고 x, y 를 반환하는 함수
def make_xy():
    df = pd.read_csv('leaf/train.csv')
    # print(df.head())

    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(df.species)

    return df.values[:, 2:], y

def make_xid():
    df = pd.read_csv('leaf/test.csv')
    # print(df.head())

    return df.values[:, 1:], df.id.values

x, y = make_xy()
# print(x.shape, y.shape)

scaler = preprocessing.StandardScaler()
x = scaler.fit_transform(x)

model = keras.Sequential(
    keras.layers.Dense(99, activation='softmax') # class 의 개수
)

model.compile(
    optimizer=keras.optimizers.SGD(.01),
    loss=keras.losses.sparse_categorical_crossentropy,
    metrics=['acc']
)

model.fit(
    x, y,
    epochs=100, # epochs=0 : 학습하지 않겠다는 뜻
    validation_split=.2,
)

x_test, leaf_ids = make_xid()
x_test = scaler.transform(x_test)


p = model.predict(x_test, verbose=0)

# print(leaf_ids.shape, p.shape) # (594,) (594, 99)

f = open('leaf/submission_keras.csv', 'w', encoding='utf-8')

f.write('id,Acer_Capillipes,Acer_Circinatum,Acer_Mono,Acer_Opalus,Acer_Palmatum,Acer_Pictum,Acer_Platanoids,Acer_Rubrum,Acer_Rufinerve,Acer_Saccharinum,Alnus_Cordata,Alnus_Maximowiczii,Alnus_Rubra,Alnus_Sieboldiana,Alnus_Viridis,Arundinaria_Simonii,Betula_Austrosinensis,Betula_Pendula,Callicarpa_Bodinieri,Castanea_Sativa,Celtis_Koraiensis,Cercis_Siliquastrum,Cornus_Chinensis,Cornus_Controversa,Cornus_Macrophylla,Cotinus_Coggygria,Crataegus_Monogyna,Cytisus_Battandieri,Eucalyptus_Glaucescens,Eucalyptus_Neglecta,Eucalyptus_Urnigera,Fagus_Sylvatica,Ginkgo_Biloba,Ilex_Aquifolium,Ilex_Cornuta,Liquidambar_Styraciflua,Liriodendron_Tulipifera,Lithocarpus_Cleistocarpus,Lithocarpus_Edulis,Magnolia_Heptapeta,Magnolia_Salicifolia,Morus_Nigra,Olea_Europaea,Phildelphus,Populus_Adenopoda,Populus_Grandidentata,Populus_Nigra,Prunus_Avium,Prunus_X_Shmittii,Pterocarya_Stenoptera,Quercus_Afares,Quercus_Agrifolia,Quercus_Alnifolia,Quercus_Brantii,Quercus_Canariensis,Quercus_Castaneifolia,Quercus_Cerris,Quercus_Chrysolepis,Quercus_Coccifera,Quercus_Coccinea,Quercus_Crassifolia,Quercus_Crassipes,Quercus_Dolicholepis,Quercus_Ellipsoidalis,Quercus_Greggii,Quercus_Hartwissiana,Quercus_Ilex,Quercus_Imbricaria,Quercus_Infectoria_sub,Quercus_Kewensis,Quercus_Nigra,Quercus_Palustris,Quercus_Phellos,Quercus_Phillyraeoides,Quercus_Pontica,Quercus_Pubescens,Quercus_Pyrenaica,Quercus_Rhysophylla,Quercus_Rubra,Quercus_Semecarpifolia,Quercus_Shumardii,Quercus_Suber,Quercus_Texana,Quercus_Trojana,Quercus_Variabilis,Quercus_Vulcanica,Quercus_x_Hispanica,Quercus_x_Turneri,Rhododendron_x_Russellianum,Salix_Fragilis,Salix_Intergra,Sorbus_Aria,Tilia_Oliveri,Tilia_Platyphyllos,Tilia_Tomentosa,Ulmus_Bergmanniana,Viburnum_Tinus,Viburnum_x_Rhytidophylloides,Zelkova_Serrata\n')



for lid, pred in zip(leaf_ids, p): # zip() : 길이기 같은 리스트를 연결
    print(lid, *pred, sep=',', file=f) # *(언패킹 연산자) : 배열의 모든 요소를 풀어서 개별 인자로 전달(공백으로 구분)

f.close()




