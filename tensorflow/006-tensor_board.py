import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, act_func=None):
    W = tf.Variable(tf.random_normal([in_size, out_size]))
    b = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, W) + b

    if act_func is None:
        return Wx_plus_b
    else:
        return act_func(Wx_plus_b)


# fake data
x = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
y = np.square(x) + np.random.normal(0, 0.05, x.shape).astype(np.float32)

xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# create layer
hidden = add_layer(x, 1, 10, tf.nn.relu)
output = add_layer(hidden, 10, 1)

# calculate loss
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - output), 1))
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()
loss_his = []

with tf.Session() as sess:
    sess.run(init)

    for i in range(1000):
        sess.run(train)

        loss_his.append(sess.run(loss))

        if i % 50 == 0:
            print(sess.run(loss))

            plt.clf()
            plt.figure(1, figsize=(10, 3))
            plt.subplot(121)
            plt.scatter(x, y, s=10, c=x, cmap='rainbow')
            plt.plot(x, sess.run(output), 'r-', lw=1)

            plt.subplot(122)
            plt.plot(loss_his)
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.ylim((0, 0.02))

            plt.pause(0.1)

    plt.pause(3)

