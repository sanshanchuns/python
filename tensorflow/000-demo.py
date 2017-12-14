import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

tf.set_random_seed(1)
np.random.seed(1)

# x = np.linspace(-1, 1, 10)[:, np.newaxis]
# index = np.random.randint(0, x.shape[0], 3)
#
# print(index)
# print(x)
# print(x[index])
# print(x[index].shape)

# x = np.random.uniform(0, 1, 1000)
# x1 = np.linspace(-1, 1, 100)
# normal = mlab.normpdf(x1, 0, 1)
# 绘制正态分布曲线
# plt.plot(x1, normal, 'r-', lw = 2)
# plt.show()

a, b = np.split(np.arange(10), [1])
print(a,b)