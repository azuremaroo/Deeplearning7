import numpy as np

arr = np.arange(1,21).reshape(5,4)
print(arr)
print('='*80)
print(arr[[1,3], [0,2]]) # fanch-indexing : 추출하고자 하는 행과 열을 배열로 전달해서 선택(추출)


