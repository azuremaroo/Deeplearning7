import numpy as np
import matplotlib.pyplot as plt

# 소숫점의 과학적 표기법의 사용 억제
np.set_printoptions(suppress=True)

x = np.arange(-5,5,0.1)
y= 1/(1+np.exp(-x)) # 시그모이드 함수 : y의 값을 0~1사이로 출력되도록

# print(x)
print(y)
plt.plot(x,y)
plt.show()