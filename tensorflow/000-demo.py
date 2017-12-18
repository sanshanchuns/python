import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)

x = [1, 2]
y = ['a', 'b']

print(x+y)

# fake data
#  np.random.seed(1)
#
# x = np.random.randn(2, 2, 1)
# y = x.reshape([1, 2, 2, 1])
# print(x.shape, y.shape)
# print(x)
# print(y)

# x = tf.Variable([[[1, 1, 1],[2, 2, 2]],
#                  [[3, 3, 3],[4, 4, 4]],
#                  [[5, 5, 5],[6, 6, 6]]])
# y = tf.reshape(x, [2, -1, 3])
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# a, b = sess.run([x, y])
#
# print(a.shape, b.shape)
#
# print(a)
# print(b)







