import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

# fake data

data = np.ones((1000, 2))

x0 = np.random.normal(2, 1, data.shape)
x1 = np.random.normal(-2, 1, data.shape)
x2 = np.random.normal(6, 1, data.shape)
x3 = np.random.normal(10, 1, data.shape)

x = np.vstack((x0, x1, x2, x3))  #(200, 2)
y = np.hstack((np.ones(1000), np.zeros(1000), 2*np.ones(1000), 3*np.ones(1000))) # (200, )

# plt.scatter(x[:, 0], x[:, 1], s=10, c=np.ones(200), cmap='rainbow')
# plt.show()

xs = tf.placeholder(tf.float32, x.shape)
ys = tf.placeholder(tf.int32, y.shape) #label 只能是int32, int64

# create graph
hidden = tf.layers.dense(xs, 10, tf.nn.relu)
output = tf.layers.dense(hidden, 4)

# loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(output), reduction_indices=[1]))
# 这里的 ys 是 (200, ), 也就是 squeeze()过的, 没有了 列的维度, labels 是不带列这个维度的
loss = tf.losses.sparse_softmax_cross_entropy(labels=ys, logits=output)
train = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    _, l, pre = sess.run([train, loss, output], feed_dict={xs: x, ys: y})

    print(l)
    if i % 10 == 0:
        plt.cla()
        plt.scatter(x[:, 0], x[:, 1], s=10, c=np.argmax(pre, axis=1), cmap='rainbow')
        plt.pause(0.1)
plt.show()





















#fake data
# n_data = np.ones((100, 2))
# x0 = np.random.normal(2*n_data, 1)      # class0 x shape=(100, 2)
# y0 = np.zeros(100)                      # class0 y shape=(100, 1)
# x1 = np.random.normal(-2*n_data, 1)     # class1 x shape=(100, 2)
# y1 = np.ones(100)                       # class1 y shape=(100, 1)
# x = np.vstack((x0, x1))  # shape (200, 2) + some noise
# y = np.hstack((y0, y1))  # shape (200, )
#
# # plot data
# plt.scatter(x[:, 0], x[:, 1], c=y, s=100, lw=0, cmap='rainbow')
# plt.show()

# data_num = 1000
#
# data = np.ones((data_num, 2)) #这里的二维是为了画图, 区分的数据本身可以是任意维度的
#
# x0 = np.random.normal(2, 1, data.shape)
# x1 = np.random.normal(-2, 1, data.shape)
# x2 = np.random.normal(6, 1, data.shape)
#
# x = np.vstack((x0, x1, x2)) #(300, 2)
# y = np.hstack((np.zeros(data_num), np.ones(data_num), 2*np.ones(data_num)))  #(300, )
#
# # plt.scatter(x[:, 0], x[:, 1], s=10, c=y, cmap='rainbow')
# # plt.show()
#
# #
# # placeholder
# xs = tf.placeholder(tf.float32, x.shape)
# ys = tf.placeholder(tf.int32, y.shape)
#
# # # create layer
# hidden = tf.layers.dense(xs, 10, tf.nn.relu)
# output = tf.layers.dense(hidden, 3)  #因此要区分三类物体，因此有3个输出[0, 1, 2]
#
# # calculate loss
# loss = tf.losses.sparse_softmax_cross_entropy(labels=ys, logits=output)
# train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
# sess = tf.Session()
# sess.run(tf.group(tf.global_variables_initializer()))
# # sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
#
# for i in range(data_num):
#     _, l, op = sess.run([train, loss, output], feed_dict={xs: x, ys: y})
#
#     if i % 50 == 0:
#         print(l)
#         prediction = sess.run(tf.argmax(op, axis=1))
#         plt.cla()
#         plt.scatter(x[:, 0], x[:, 1], s=10, c=prediction, cmap='rainbow')
#         plt.pause(0.1)
# plt.pause(3)
