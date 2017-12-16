import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

N_SAMPLES = 20
N_HIDDEN = 300
LR = 0.01

tf.set_random_seed(1)
np.random.seed(1)

#fake train data
x = np.linspace(-1, 1, N_SAMPLES)[:, np.newaxis]
y = np.power(x, 2) + 0.3*np.random.randn(N_SAMPLES)[:, np.newaxis]

#fake test data
x_test = x.copy()
y_test = np.power(x_test, 2) + 0.3*np.random.randn(N_SAMPLES)[:, np.newaxis]

#placeholder
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(dtype=tf.float32, shape=[None, 1])
tf_is_training = tf.placeholder(tf.bool, None)

#overfit layer
o1 = tf.layers.dense(xs, N_HIDDEN, tf.nn.relu)
#再加一层用来进一步加大 overfit 的程度, 没有 dropout
o2 = tf.layers.dense(o1, N_HIDDEN, tf.nn.relu)
o_out = tf.layers.dense(o2, 1)
o_loss = tf.losses.mean_squared_error(ys, o_out)
o_train = tf.train.AdamOptimizer(LR).minimize(o_loss)

#dropout overfit layer
d1 = tf.layers.dense(xs, N_HIDDEN, tf.nn.relu)
d1 = tf.layers.dropout(d1, rate=0.5, training=tf_is_training)
#再加一层用来进一步加大 overfit 的程度,  dropout
d2 = tf.layers.dense(d1, N_HIDDEN, tf.nn.relu)
d2 = tf.layers.dropout(d2, rate=0.5, training=tf_is_training)
d_out = tf.layers.dense(d1, 1)
d_out = tf.layers.dropout(d_out, rate=0.5, training=tf_is_training)
d_loss = tf.losses.mean_squared_error(ys, d_out)
d_train = tf.train.AdamOptimizer(LR).minimize(d_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

losses_train = []
losses_test = []

for t in range(500):
    # train, set is_training=True
    sess.run([o_train, d_train], {xs: x, ys: y, tf_is_training: True})

    if t % 10 == 0:
        # plotting
        plt.cla()
        o_loss_, d_loss_, o_out_, d_out_ = sess.run(
            [o_loss, d_loss, o_out, d_out],
            {xs: x_test, ys: y_test, tf_is_training: False}
            # test, set is_training=False
        )
        plt.scatter(x, y, c='magenta', s=50, alpha=0.3, label='train')
        plt.scatter(x_test, y_test, c='cyan', s=50, alpha=0.3, label='test')
        plt.plot(x_test, o_out_, 'r-', lw=3, label='overfitting')
        plt.plot(x_test, d_out_, 'b--', lw=3, label='dropout(50%)')
        plt.text(0, -1.2, 'overfitting loss=%.4f' % o_loss_,
                 fontdict={'size': 20, 'color':  'red'})
        plt.text(0, -1.5, 'dropout loss=%.4f' % d_loss_,
                 fontdict={'size': 20, 'color': 'blue'})
        plt.legend(loc='upper left')
        plt.ylim((-2.5, 2.5))
        plt.pause(0.1)
plt.show()

# for i in range(500):
#     _, l, op = sess.run([train0, loss0, output0],
# feed_dict={xs: x, ys: y, tf_is_training: True})
#     _, l_test, op_test = sess.run([train1, loss1, output1],
# feed_dict={xs: x_test, ys: y_test, tf_is_training: False})
#     losses_train.append(l)
#     losses_test.append(l_test)
#
#     if i % 10 == 0:
#         plt.cla()
#         plt.scatter(x, y, c=x, cmap='rainbow')
#         plt.plot(x, op, 'r-', lw=1)
#         plt.plot(x_test, op_test, 'b-', lw=1)
#         plt.pause(0.1)
# plt.pause(3)

# plt.cla()
# plt.plot(losses_train, label='train')
# plt.plot(losses_test, label='test')
# plt.legend(loc='best')
# plt.xlabel('steps')
# plt.ylabel('losses')
# plt.show()


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
#         plt.plot(x, sess.run(output,
# feed_dict={xs: x, ys: y}), 'r-', lw=1)
#         plt.pause(0.1)
#
# plt.pause(3)