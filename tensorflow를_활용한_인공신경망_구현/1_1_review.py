# 1_1_review.py
import numpy as np

# 퀴즈
# 0 ~ 24까지 한 줄에 5개씩 정수를 출력하세요 (2가지)
#      0  1  2  3  4
# ------------------
# 0 |  0  1  2  3  4
# 1 |  5  6  7  8  9    1 * 5 + 2 = 7
# 2 | 10 11 12 13 14
# 3 | 15 16 17 18 19
# 4 | 20 21 22 23 24    4 * 5 + 1 = 21

#       0  1  2  3  4
# -------------------
#  0 |  0  1  2  3  4
#  5 |  5  6  7  8  9    5 + 2 = 7
# 10 | 10 11 12 13 14
# 15 | 15 16 17 18 19
# 20 | 20 21 22 23 24    20 + 1 = 21

for i in range(5):
    for j in range(5):
        print(i*5 + j, end=' ')
    print()
print()

for i in range(0, 25, 5):
    for j in range(5):
        print(i + j, end=' ')
    print()
print()

for i in range(25):
    print(i, end=' ')

    # if i == 4 or i == 9 or i == 14 or i == 19 or i == 24:
    # if i % 10 == 4 or i % 10 == 9:
    if i % 5 == 4:
        print()
print()

a = np.arange(12)
print(a)

print(a + 1)            # broadcast
print(1 + a)
print(a + a)            # vector
print(np.sin(a))        # universal

b = a.reshape(-1, 4)
print(b)

a[0] = 99

# 퀴즈
# 2차원 배열을 1차원으로 바꾸세요 (3가지)
print(b.reshape(12))
print(b.reshape(b.size))
print(b.reshape(-1))
print()

b0 = np.arange(3)
b1 = np.arange(6)
b2 = np.arange(3).reshape(1, 3)
b3 = np.arange(6).reshape(2, 3)
b4 = np.arange(3).reshape(3, 1)

# print(b0 + b1)
print(b0 + b2)
print(b0 + b3)      # broadcast + vector

# (1, 3)
# (2, 3)
# ------
# (2, 3)

print(b0 + b4)

# (1, 3)
# (3, 1)
# ------
# (3, 3)

# print(b1 + b2)      # (1, 6) + (1, 3)
# print(b1 + b3)      # (1, 6) + (2, 3)
print(b1 + b4)      # (1, 6) + (3, 1)

print(b2 + b3)      # (1, 3) + (2, 3)
print(b2 + b4)      # (1, 3) + (3, 1)

# print(b3 + b4)      # (2, 3) + (3, 1)
print()

# 퀴즈
# 2차원 배열 b에서 마지막 요소를 출력하세요
print(b)
print(b[-1])
print(b[-1][-1])
print(b[-1, -1])        # fancy indexing

# 퀴즈
# 2차원 배열 b를 거꾸로 출력하세요
print(b[::-1])
print(b[::-1][::-1])
print(b[::-1, ::-1])
print()

# 퀴즈
# 테두리를 1로 채우세요
z = np.zeros([5, 5], dtype=np.int32)

# z[0] = 1
# z[-1] = 1
# z[0, :] = 1
# z[-1, :] = 1
z[[0, -1], :] = 1

# z[:, 0] = 1
# z[:, -1] = 1
z[:, [0, -1]] = 1

print(z, end='\n\n')

# 퀴즈
# 가운데를 2로 채우세요
z[1:-1, 1:-1] = 2
print(z, end='\n\n')

# 퀴즈
# 대각선을 3으로 채우세요
# z[0, 0] = 3
# z[1, 1] = 3
# z[2, 2] =, [0, 1, 2 3
# z[[0, 1, , [0, 1, 22]] = 3
# z[[0, 1, 2], [0, 1, 2]] = 3
z[range(5), range(5)] = 3
# z[range(5), list(reversed(range(5)))] = 3
print(z, end='\n\n')
# print(z[[0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4]])


