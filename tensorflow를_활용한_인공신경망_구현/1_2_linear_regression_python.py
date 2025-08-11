# 1_2_linear_regression_python.py
import matplotlib.pyplot as plt

def cost(x, y, w):
    c = 0
    for i in range(len(x)):
        hx = w * x[i]
        c += (hx - y[i]) ** 2


    return c / len(x)

def gradient_descent(x, y, w):
    c = 0
    for i in range(len(x)):
        hx = w * x[i]
        c += (hx - y[i]) * x[i]
        # c += (hx - y[i]) ** 2
        # c += (w * x[i] - y[i]) ** 2
        # c += 2 * (w * x[i] - y[i]) ** (2-1) * (w * x[i] - y[i])미분
        # c += 2 * (w * x[i] - y[i]) * x[i]
        # c += (w * x[i] - y[i]) * x[i]
        # c += (hx - y[i]) * x[i]

    return c / len(x)

def show_cost():
    # y = ax + b
    # hx = wx + b
    #      1    0
    x = [1, 2, 3]
    y = [1, 2, 3]

    # print(cost(x, y, w=0))
    # print(cost(x, y, w=1))
    # print(cost(x, y, w=2))

    for i in range(-30, 50):
        w = i / 10
        c = cost(x, y, w)

        plt.plot(w, c, 'ro')
    plt.show()

def show_gradient():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = 5
    for i in range(10):
        c = cost(x, y, w)
        g = gradient_descent(x, y, w)
        w -= 0.1 * g

        print(i, c)

    # 퀴즈
    # x가 5와 7일 때의 결과를 예측하세요
    print('5 :', w * 5)
    print('7 :', w * 7)


# show_cost()
show_gradient()


# 미분 : 기울기, 순간변화량
#       x축으로 1만큼 움직였을 때, y축으로 움직인 거리

# y = 7         7=1, 7=2, 7=3
# y = x         1=1, 2=2, 3=3
# y = 2x        2=1, 4=2, 6=3
# y = (x+1)     2=1, 3=2, 4=3
# y = xz

# y = x^2       1=1, 4=2, 9=3
#     2 * x^(2-1) * x미분 = 2x
# y = (x+1)^2
#     2 * (x+1)^(2-1) * (x+1)미분 = 2(x+1)

