import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def add_layer(inputs, in_size, out_size, act_func=None):
    W = tf.Variable(tf.random_normal([in_size, out_size]))
    b = tf.Variable(tf.zeros([1, out_size]))
    Wx_plus_b = tf.matmul(inputs, W) + b
    if act_func is None:
        outputs = Wx_plus_b
    else:
        outputs = act_func(Wx_plus_b)
    return outputs

def compute_accuracy(x, y):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: x})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: x, ys: y})
    return result

# xs = tf.placeholder(tf.float32, [None, 1]) #个数没有限制，但必须是一维的
# ys = tf.placeholder(tf.float32, [None, 1])

xs = tf.placeholder(tf.float32, [None, 28*28])
ys = tf.placeholder(tf.float32, [None, 10])

# x = np.linspace(-1, 1, 300)[:, np.newaxis]
# y = np.square(x) + np.random.normal(0, 0.05, x.shape)

#计算预测值
# hidden = add_layer(xs, 1, 10, tf.nn.relu)
# output = add_layer(hidden, 10, 1)

prediction = add_layer(xs, 28*28, 10, tf.nn.softmax)
loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), 1))
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - output), 1))
# train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    # sess.run(train, feed_dict={xs: x, ys: y})
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train, feed_dict={xs: batch_x, ys: batch_y})

    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))