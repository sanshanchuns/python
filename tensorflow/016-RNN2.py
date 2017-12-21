import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import numpy as np
import pkg_dataset.dataset_captcha as dataset

TIME_STEP = 60
INPUT_SIZE = 160
BATCH_SIZE = 20
LR = 0.001
MAX_CAPTCHA = 4
CHAR_SET_LEN = 63  #(10 + 26 + 26 + 1)

xs = tf.placeholder(tf.float32, [None, TIME_STEP*INPUT_SIZE])
xs_2d = tf.reshape(xs, [-1, TIME_STEP, INPUT_SIZE])
ys = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])

outputs, _ = tf.nn.dynamic_rnn(
    cell=tf.nn.rnn_cell.BasicLSTMCell(64),
    inputs=xs_2d,
    initial_state=None,
    dtype=tf.float32,
    time_major=False,
)
output = tf.layers.dense(outputs[:, -1, :], MAX_CAPTCHA*CHAR_SET_LEN)

loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=ys, logits=output)
train = tf.train.AdamOptimizer(LR).minimize(loss)
accuracy = tf.metrics.accuracy(labels=tf.argmax(ys, 1), predictions=tf.argmax(output, 1))[1]

sess = tf.Session()
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
saver = tf.train.Saver()

for i in range(10000):
    b_x, b_y = dataset.next_batch(BATCH_SIZE)
    sess.run([train], feed_dict={xs: b_x, ys: b_y})

    if i % 50 == 0:
        t_x, t_y = dataset.next_batch(BATCH_SIZE)
        l, op, ac = sess.run([loss, output, accuracy], feed_dict={xs: t_x, ys: t_y})
        print(i, l, ac)
        op_index = np.argmax(op[0, :].reshape(MAX_CAPTCHA, CHAR_SET_LEN), 1)
        print(dataset.index2text(op_index))
        print(dataset.vec2text(t_y[0, :]))
        if ac > 0.4 or l < 0.03:
            print('save')
            saver.save(sess, './rnn_params/', write_meta_graph=False)