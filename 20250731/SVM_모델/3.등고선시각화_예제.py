import numpy as np
import matplotlib.pyplot as plt

print(np.inf)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
x1 = np.linspace(1, 5, 10)
y2 = np.linspace(6, 10, 10)

xx1, yy2 = np.meshgrid(x1, y2)
# print(xx1)
# print(yy2)

Z = xx1**2 + yy2**2

plt.contour(xx1, yy2, Z, levels=[30,50,100,150], colors='black')
# plt.contourf(xx1, yy2, Z, levels=[30,50,100,150], colors=['r','g','b'])
plt.contourf(xx1, yy2, Z, levels=[30,50,100,150], cmap='winter', alpha=0.3)
plt.show()