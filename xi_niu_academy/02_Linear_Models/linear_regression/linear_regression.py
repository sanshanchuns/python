# import numpy as np
# import matplotlib.pyplot as plt
#
# from sklearn.linear_model import LinearRegression
#
# data = np.loadtxt('linear_regression_data1.txt', delimiter=',')
# # print(data)
#
# X = np.c_[np.ones(data.shape[0]), data[:, 0]]
# y = np.c_[data[:, 1]]
#
# # plt.scatter(X[:, 1], y, s=50, c='r', marker='x', lw=1)
# # plt.show()
#
# # plt.scatter(data[:, 0], data[:, 1], s=50, c='r', marker='x', lw=1)
# # plt.show()
#
# # print(y.shape)
# # print(y.size)
# # print(X.dot([[0, 0]]))
# # print(np.array([0, 0]).shape)
# # print(np.array([[0], [0]]).shape)
# # print(np.array([[0, 0]]).shape)
# # print(np.array([[[0, 0]]]).shape)
#
# # print(X.dot([[0], [0]]))
#
# def loss(X, y, theta=[[0], [0]]):
#     m = y.size
#     l = 0
#     predict = X.dot(theta)  # (97, 2) . (2, 1)  = (97, 1)
#     l = 1.0/(2*m)*(np.sum(np.square(predict-y)))
#     return l
#
# # print(loss(X, y))
#
# def gd(X, y, theta=[[0], [0]], alpha = 0.01, num_iters = 1500):
#     m = y.size
#     l_his = np.zeros(num_iters)
#
#     for iter in range(num_iters):
#         predict = X.dot(theta) # (97, 2) . (2, 1)  = (97, 1)
#         theta -= alpha*(1.0/m)*(X.T.dot(predict-y)) # (2, 97).(97.1) = (2, 1)
#         l_his[iter] = loss(X, y, theta)
#     return theta, l_his
#
# theta, l_his = gd(X, y)  # (2, 1)
# # print('theta: ', theta.flatten())  # theta.ravel() VS theta.flatten()
#
# # plt.plot(l_his)
# # plt.ylabel('Cost')
# # plt.xlabel('Iterations')
# # plt.show()
#
# xx = np.arange(5, 23)
# yy = theta[0] + theta[1]*xx
#
# plt.scatter(X[:, 1], y, s=50, c='r', marker='x', lw=1)
# plt.plot(xx, yy, label='manual')
#
# # 对比sciki
# regr = LinearRegression()
# regr.fit(X[:, 1].reshape(-1, 1), y.ravel())
# plt.plot(xx, regr.intercept_ + regr.coef_*xx, label='sciki')
#
# plt.legend(loc='best')
# # plt.show()
#
# # 顺便预测一下 35000 与 70000 的结果
#
# print(theta.T.dot([1, 3.5]))
#
# print(np.array([[1, 3.5]]).dot(theta))
#
# print(theta.T.dot([1, 7]))
#
#

#完成. 重新操练

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = np.loadtxt('linear_regression_data1.txt', delimiter=',')

# np.c_ 列合并  np.r_ 行合并

X = np.c_[np.ones(data.shape[0]), data[:, 0]]  #转换为列的形式
y = np.c_[data[:, 1]]
# y = data[:, 1].reshape(-1, 1)


def loss(X, y, theta=[[0], [0]]):
    m = y.size
    l = 1.0/(2*m)*np.sum(np.square(X.dot(theta)-y))
    # 这里有的 predict = X.dot(theta)
    return l


def gd(X, y, theta=[[0], [0]], alpha=0.01, num_iter=1500):
    m = y.size
    l_his = []

    for i in range(num_iter):
        theta -= alpha* (1.0/m)*X.T.dot(X.dot(theta)-y)
        l = loss(X, y, theta)
        l_his.append(l)
    return theta, l_his

theta, l_his = gd(X, y)

plt.ion()

plt.scatter(X[:, 1], y.flatten(), s=30, c='r', marker='x', lw=1)

xx = np.arange(5, 23)
yy = theta[1]*xx + theta[0]

plt.plot(xx, yy, label='manual')

regr = LinearRegression()
# 这个地方 X 需要转换成多行的形式， y标签需要转换成 单行的形式
regr.fit(X[:, 1].reshape(-1, 1), y.flatten())
yy_regr = regr.intercept_ + regr.coef_*xx

plt.plot(xx, yy_regr, label='sciki')
plt.show()

plt.ioff()

# 预测值 形式是 X.theta  或者 theta.T.X

print(theta.T.dot([1, 3.5])* 10000)
print(np.array([1, 7]).dot(theta)* 10000)