
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# def add_layer(inputs, in_size, out_size, act_func=None):
#     W = tf.Variable(tf.random_normal([in_size, out_size]))
#     b = tf.Variable(tf.zeros([1, out_size]) + 0.1)
#     Wx_plus_b = tf.matmul(inputs, W) + b  #矩阵相乘顺序不能颠倒
#
#     if act_func is None:
#         return Wx_plus_b
#     else:
#         return act_func(Wx_plus_b)
#
#
# # x = np.linspace(-1, 1, 300)[:, np.newaxis]
# x = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
# # y = np.square(x) - 0.5 + np.random.normal(0, 0.05, x.shape)
# y = np.square(x) - 0.5 + np.random.normal(0, 0.05, x.shape).astype(np.float32)
#
# # xs = tf.placeholder(tf.float32, [None, 1])  #如果这里使用了占位符，那么上面的数据定义就不需要类型了
# # ys = tf.placeholder(tf.float32, [None, 1])
#
# hidden = add_layer(x, 1, 10, tf.nn.relu) #torch.nn.Linear(1, 10)
# output = add_layer(hidden, 10, 1)  #torch.nn.Linear(10, 1)
#
# # loss = tf.reduce_mean(tf.square(y - output))
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - output), reduction_indices=[1]))
# train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#
#     for step in range(1000):
#         sess.run(train)
#
#         if step % 50 == 0:
#
#             plt.cla()
#             plt.scatter(x, y, s=10, c=x, cmap='rainbow')
#             plt.plot(x, sess.run(output), 'r-', lw=1)
#             plt.pause(0.1)
#
#             print(sess.run(loss))
#
# plt.pause(3)

def add_layer(inputs, in_size, out_size, act_func=None):
    W = tf.Variable(tf.random_normal([in_size, out_size]))
    b = tf.Variable(tf.zeros([1, out_size]))
    # Wx_plus_b = tf.matmul(inputs, W) + b
    Wx_plus_b = tf.add(tf.matmul(inputs, W), b)  #一样的表达

    if act_func is None:
        return Wx_plus_b
    else:
        return act_func(Wx_plus_b)

#0. create linear func
x = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
y = np.square(x) + np.random.normal(0, 0.05, x.shape).astype(np.float32)

#1. create layer
hidden = add_layer(x, 1, 10, tf.nn.relu)  #这里只是给一个函数名，即函数指针，不能给函数调用
output = add_layer(hidden, 10, 1)

#2. calculate loss
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - output), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(1000):
        sess.run(train)

        if i % 50 == 0:

            plt.cla()
            plt.scatter(x, y, s=10, c=x, cmap='rainbow')
            plt.plot(x, sess.run(output), 'r-', lw=1)
            plt.pause(0.1)

            print(sess.run(loss))

plt.pause(3)