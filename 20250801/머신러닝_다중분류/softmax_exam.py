import numpy as np


# 소숫점의 과학적 표기법의 사용 억제
np.set_printoptions(suppress=True)

mylist = [-6.5,1.03,5.16,-2.73,3.34,0.33,-0.63]
arr = np.array(mylist)

exp_a = np.exp(arr)
sum_exp_a = np.sum(exp_a)

y = exp_a / sum_exp_a
print(y)
output = np.round(y, decimals=3)
print(output)
print(np.argmax(output)) # np.argmax() ==> 가장 큰 값의 인덱스 반환
class_name = ['가','나','다','라']
print(class_name[np.argmax(output)]) # 사용자에게 인덱스 이름으로 보이도록 처리