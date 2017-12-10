import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, n_layer, act_func=None):

    layer_name = 'layer%s' % n_layer

    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            W = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/weights', W)
        with tf.name_scope('biases'):
            b = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', b)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, W) + b

        if act_func is None:
            outputs = Wx_plus_b
        else:
            outputs = act_func(Wx_plus_b)

        tf.summary.histogram(layer_name + '/outputs', outputs)

    return outputs


# fake data
x = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
y = np.square(x) + np.random.normal(0, 0.05, x.shape).astype(np.float32)

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# create layer

hidden = add_layer(x, 1, 10, 1, tf.nn.relu)
output = add_layer(hidden, 10, 1, 2)

# calculate loss
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - output), 1))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()
loss_his = []

sess = tf.Session()

merged = tf.summary.merge_all()

writer = tf.summary.FileWriter('logs/', sess.graph)

sess.run(init)
for i in range(1000):
    sess.run(train)

    loss_his.append(sess.run(loss))

    if i % 50 == 0:
        rs = sess.run(merged)
        writer.add_summary(rs, i)

        # plt.clf()
        # plt.figure(1, figsize=(10, 3))
        # plt.subplot(121)
        # plt.scatter(x, y, s=10, c=x, cmap='rainbow')
        # plt.plot(x, sess.run(output), 'r-', lw=1)
        #
        # plt.subplot(122)
        # plt.plot(loss_his)
        # plt.xlabel('Steps')
        # plt.ylabel('Loss')
        # plt.ylim((0, 0.02))
        #
        # plt.pause(0.1)

