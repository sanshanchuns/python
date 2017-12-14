import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


tf.set_random_seed(1)
np.random.seed(1)

#fake data
x = np.linspace(-1, 1, 300)[:, np.newaxis]
y = np.power(x, 2) + np.random.normal(0, 0.05, x.shape)

np.random.randn()

#placeholder
xs = tf.placeholder(tf.float32, x.shape)
ys = tf.placeholder(dtype=tf.float32, shape=y.shape)

#create layer
hidden = tf.layers.dense(xs, 10, tf.nn.relu)
output = tf.layers.dense(hidden, 1)

#calculate loss
loss = tf.losses.mean_squared_error(ys, output)
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    sess.run(train, feed_dict={xs: x, ys: y})

    if i % 50 == 0:
        print(sess.run(loss, feed_dict={xs: x, ys: y}))

        plt.cla()
        plt.scatter(x, y, c=x, s=10, cmap='rainbow')
        plt.plot(x, sess.run(output, feed_dict={xs: x, ys: y}), 'r-', lw=1)
        plt.pause(0.1)
plt.pause(3)


# tf.set_random_seed(1)
# np.random.seed(1)
#
# #fake data
# x = np.linspace(-1, 1, 300)[:, np.newaxis]
# y = np.power(x, 2) + np.random.normal(0, 0.05, x.shape)
#
# #placeholder
# xs = tf.placeholder(tf.float32, x.shape)
# ys = tf.placeholder(tf.float32, y.shape)
#
# #create layer
# hidden = tf.layers.dense(xs, 10, tf.nn.relu)
# output = tf.layers.dense(hidden, 1)
#
# #create loss
# loss = tf.losses.mean_squared_error(ys, output)
# train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# for step in range(1000):
#     sess.run(train, feed_dict={xs: x, ys: y})
#
#     if step % 50 == 0:
#         print(sess.run(loss, feed_dict={xs: x, ys: y}))
#
#         plt.cla()
#         plt.scatter(x, y, s=10, c=x, cmap='rainbow')
#         plt.plot(x, sess.run(output, feed_dict={xs: x, ys: y}), 'r-', lw=1)
#         plt.pause(0.1)
#
# plt.pause(3)