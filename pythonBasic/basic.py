# l = [
#     ['test', 'hello', 'world', 'sm', 'philpbin', 'english'],
#     ['test2', 'hello2', 'world2', 'sm2', 'philpbin2', 'english2'],
#     ]
#
# for i, data in enumerate(l, 0):
#     print(i, data)

import numpy as np

# data = np.loadtxt('linear_regression_data1.txt', delimiter=',')
# print(data[:, 0])
# print(np.c_[np.ones(data.shape[0]), data[:, 0]])

# print(data[:,1])
# print(np.c_[data[:,1]])

a = np.ones([2, 3])
# b = np.ones([2, 3])
# print(a.size)

a = np.array([[1, 2],
              [3, 4]])
b = np.array([[5], [6]])

# print(a.dot(b))

print(a)

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(a.dot(b))


# print(np.ones([2, 1]))
# print(np.ones((2, 1)))

# print(a.T.shape)
#
# m = np.matrix('[1, 2; 3, 4]')
# print(m.T)


