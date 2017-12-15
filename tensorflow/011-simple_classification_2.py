import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.examples.tutorials.mnist import input_data


# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# def add_layer(inputs, in_size, out_size, activation_function=None,):
#     # add one more layer and return the out_size]) + 0.1,)
#     Wx_plus_b = tf.matm output of this layer
#     Weights = tf.Variable(tf.random_normal([in_size, out_size]))
#     biases = tf.Variable(tf.zeros([1,ul(inputs, Weights) + biases
#     if activation_function is None:
#         outputs = Wx_plus_b
#     else:
#         outputs = activation_function(Wx_plus_b,)
#     return outputs
#
# def compute_accuracy(v_xs, v_ys):
#     global prediction
#     y_pre = sess.run(prediction, feed_dict={xs: v_xs})
#     correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
#     return result

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])  # 28x28, 输入就是 784 个维度
ys = tf.placeholder(tf.float32, [None, 10])  # 输出就是 10 个维度

# add output layer
# prediction = add_layer(xs, 784, 10,  activation_function=tf.nn.softmax)
hidden = tf.layers.dense(xs, 784, tf.nn.relu)
output = tf.layers.dense(hidden, 10, tf.nn.softmax)

# the error between prediction and real data
# loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(output), axis=1))
loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=ys, logits=output)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(100):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    _, l, op = sess.run([train_step, loss, output], feed_dict={xs: batch_xs, ys: batch_ys})
    if i == 99:
        print(l)
        print(batch_ys.shape)
        print(op)
        # print(np.argmax(op, axis=1))
        # print(np.argmax(batch_ys, axis=1))