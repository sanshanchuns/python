import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
BATCH_SIZE = 256
EPOCH = 5
LR = 0.01

mnist = input_data.read_data_sets('./mnist', one_hot=True)

xs = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])

l1 = tf.layers.dense(xs, 256, tf.nn.sigmoid)
l2 = tf.layers.dense(l1, 128, tf.nn.sigmoid)
l3 = tf.layers.dense(l2, 256, tf.nn.sigmoid)
output = tf.layers.dense(l3, 784, tf.nn.sigmoid)

loss = tf.losses.mean_squared_error(labels=xs, predictions=output)
train = tf.train.AdamOptimizer(LR).minimize(loss)
accuracy = tf.metrics.accuracy(labels=tf.argmax(xs, 1), predictions=tf.argmax(output, 1))[1]

sess = tf.Session()
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

for i in range(EPOCH):
    b_x, _ = mnist.train.next_batch(BATCH_SIZE)
    sess.run([train], feed_dict={xs: b_x})

    l, ac, op = sess.run([loss, accuracy, output], feed_dict={xs: b_x})
    print(l, ac)

plt.figure(1, figsize=(12, 5))
for index in range(1, 10):
    image = b_x[index].reshape(28, 28)
    plt.subplot('29%d' % index)
    plt.imshow(image, cmap='rainbow')

for index in range(10, 19):
    image = op[index].reshape(28, 28)
    plt.subplot(2, 9, int('%d' % index))
    plt.imshow(image, cmap='rainbow')

plt.show()
