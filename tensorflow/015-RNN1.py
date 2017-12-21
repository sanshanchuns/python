import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

TIME_STEP = 28
INPUT_SIZE = 28
MAX_CAPTCHA = 1
CHAR_SET_LENGTH = 10
LR = 0.01
BATCH_SIZE = 32

mnist = input_data.read_data_sets('./mnist', one_hot=True)

xs = tf.placeholder(tf.float32, [None, TIME_STEP*INPUT_SIZE])
xs_2d = tf.reshape(xs, [-1, TIME_STEP, INPUT_SIZE])
ys = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LENGTH])

# time_major
# If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
# If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
outputs, _ = tf.nn.dynamic_rnn(
    cell=tf.nn.rnn_cell.BasicLSTMCell(64), #输出64
    inputs=xs_2d,
    initial_state=None,
    time_major=False,
    dtype=tf.float32,
)
output = tf.layers.dense(outputs[:, -1, :], MAX_CAPTCHA*CHAR_SET_LENGTH)

loss = tf.losses.softmax_cross_entropy(onehot_labels=ys, logits=output)
train = tf.train.AdamOptimizer(LR).minimize(loss)
accuracy = tf.metrics.accuracy(labels=tf.argmax(ys, 1), predictions=tf.argmax(output, 1))[1]
accuracy_his = []

sess = tf.Session()
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

for i in range(1000):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    sess.run([train], feed_dict={xs: b_x, ys: b_y})

    if i % 50 == 0:
        t_x, t_y = mnist.test.next_batch(BATCH_SIZE)
        l, op, ac = sess.run([loss, output, accuracy], feed_dict={xs: t_x, ys: t_y})
        accuracy_his.append(ac)

plt.plot(accuracy_his)
plt.ylim((0.5, 1))
plt.show()













# mnist = input_data.read_data_sets('./mnist', one_hot=True)

# xs = tf.placeholder(tf.float32, [None, TIME_STEP*INPUT_SIZE])
# xs_2d = tf.reshape(xs, [-1, TIME_STEP, INPUT_SIZE])
# ys = tf.placeholder(tf.float32, [None, CHAR_SET_LENGTH])
#
# outputs, _ = tf.nn.dynamic_rnn(
#     tf.nn.rnn_cell.BasicLSTMCell(64),
#     xs_2d,
#     initial_state=None,
#     dtype=tf.float32,
#     time_major=False,
# )
#
# # outputs = (?, 28, 64), 取所有行, 小段中的最后一行，所有列
# # 因为时间序列是从第一行到最后一行，因此最后一行才是输出结果，包含之前所有的分析
# output = tf.layers.dense(outputs[:, -1, :], 1*CHAR_SET_LENGTH)
#
# loss = tf.losses.softmax_cross_entropy(onehot_labels=ys, logits=output)
# train = tf.train.AdamOptimizer(LR).minimize(loss)
# accuracy = tf.metrics.accuracy(labels=tf.argmax(ys, 1), predictions=tf.argmax(output, 1))[1]
#
# sess = tf.InteractiveSession()
# sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
#
# for i in range(1000):
#     b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
#     sess.run([train], feed_dict={xs: b_x, ys: b_y})
#
#     if i % 50 == 0:
#         t_x, t_y = mnist.test.next_batch(BATCH_SIZE)
#         l, op, ac = sess.run([loss, output, accuracy], feed_dict={xs: t_x, ys: t_y})
#         print(l, ac)
#
#         print(np.argmax(t_y, 1))
#         print(np.argmax(op, 1))