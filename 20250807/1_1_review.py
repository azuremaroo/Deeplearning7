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

print('a + 1 : ', a + 1) # numpy broadcasting 연산 (자주 사용)
print('1 + a : ', 1 + a) # 덧셈의 교환 법칙 적용됨
print('a + a : ', a + a) # vector 연산 (배열 연산)
print('np.sin(a) : ', np.sin(a)) # universal 함수 : 배열 요소에 모두 적용되는 함수

b = a.reshape(3, -1) # 배열의 요소 만큼 확장해야 reshape 가능
print(b)

# 퀴즈 : 2차원 배열을 1차원으로 바꾸세요
# print(b.reshape(12))
print(b.reshape(-1))
# print(b.reshape(b.size))
# print(b.flatten())
print()

# 파이썬의 deepcopy
# 1. [].copy() 함수
# 2. [:] 슬라이싱

# 배열 연산
b0 = np.arange(3)
b1 = np.arange(6)
b2 = np.arange(3).reshape(1, 3)
b3 = np.arange(6).reshape(2, 3)
b4 = np.arange(3).reshape(3, 1)

# print(b0 + b1) # 배열 크기가 다르면 계산할 수 없음
# print(b0 + b2) # 차원이 다르면 높은 차원으로 승격
# print(b0 + b3) # 차원이 다르면 높은 차원으로 승격

print(b2 + b3) # 행은 행끼리(1, 2) 열은 열끼리(3, 3) 연산하므로 가능
# (1, 3)
# (2, 3)
# ------
# (2, 3) [[0 2 4] ==> broadcasting + vector 연산 적용
#         [3 5 7]]

print(b0 + b4)
# (1, 3)
# (3, 1)
# ------
# (3, 3) [[0 1 2]
#         [1 2 3]
#         [2 3 4]]

# 행과 열에 vector(각 배열의 행과 열이 같은 경우) 연산이나
# broadcasting 연산(한쪽의 행이나 열이 1인 경우)이 가능해야 서로 다른 배열의 연산 가능
# print(b1 + b2) # (1, 6) + (1, 3) 연산 불가
# print(b1 + b3) # (1, 6) + (2, 3) 연산 불가
print('b1 + b4 : ', b1 + b4) # (1, 6) + (3, 1) 가능 (3, 6)
print('b2 + b3 : ', b2 + b3) # (1, 3) + (2, 3) 가능 (2, 3)
print('b2 + b4 : ', b2 + b4) # (1, 3) + (3, 1) 가능 (3, 3)
# print(b3 + b4) # (2, 3) + (3, 1) 연산 불가

# fancy indexing, slicing, 인덱싱 배열 연습
# fancy indexing : 차원의 개수만큼 , 로 구분
# 퀴즈 : 2차원 배열 b에서 마지막 요소를 출력하세요
print('b 의 마지막 요소 : ', b[-1, -1]) # ([][] 은 배열 연산자를 2번 호출하므로 비추)

# 퀴즈 : 2차원 배열 b를 거꾸로 출력하세요
print('b를 거꾸로 출력 : ', b[::-1, ::-1])

z = np.zeros([5, 5], dtype=np.int32)
print(z, end='\n\n')

# 인덱스 배열
# 퀴즈 : 테두리를 1로 채우세요
z[:, [0, -1]] = 1
z[[0, -1], :] = 1
print(z, end='\n\n')

# 퀴즈 : 테두리를 제외한 나머지를 2로 채우세요
z[1:-1, 1:-1] = 2
print(z, end='\n\n')

# 퀴즈 : 대각선을 3으로 채우세요
# z[[0,1,2,3,4], [0,1,2,3,4]] = 3
z[range(5), range(5)] = 3
# z[[4,3,2,1,0], [0,1,2,3,4]] = 3
z[range(5), range(5)] = 3
print(z, end='\n\n')
print()

