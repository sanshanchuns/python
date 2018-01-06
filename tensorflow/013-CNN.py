import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 50
BATCH_SIZE_TEST = 10
LR = 0.01
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
CHAR_SET_LEN = 10

tf.set_random_seed(1)
np.random.seed(1)

mnist = input_data.read_data_sets('mnist', one_hot=True)

xs = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH]) # 28*28
xs_4d = tf.reshape(xs, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1]) #这里的 -1 表示未知的batch数量，即每批图片的数量
ys = tf.placeholder(tf.float32, [None, 1*CHAR_SET_LEN]) #一个数字，十种可能

# con1 = tf.layers.conv2d(image, 16, 5, 1, 'same', activation=tf.nn.relu) # 28*28*16
con1 = tf.layers.conv2d(inputs=xs_4d, filters=16, kernel_size=5, strides=1, padding='valid', activation=tf.nn.relu) # 24*24*16
con1 = tf.layers.dropout(con1, rate=0.5)
p1 = tf.layers.max_pooling2d(inputs=con1, pool_size=2, strides=2, padding='same') # 12*12*16
# con2 = tf.layers.conv2d(p1, 32, 5, 1, 'same', activation=tf.nn.relu) #14*14*32
con2 = tf.layers.conv2d(p1, 32, 5, 1, 'valid', activation=tf.nn.relu) #8*8*32
con2 = tf.layers.dropout(con2, rate=0.5)
p2 = tf.layers.max_pooling2d(con2, 2, 2) #4*4*32
# 每次识别一个数字，每个数字有10种可能
output = tf.layers.dense(tf.reshape(p2, [-1, 4*4*32]), 1*CHAR_SET_LEN) # => (-1, 1*10)

loss = tf.losses.softmax_cross_entropy(onehot_labels=ys, logits=output)
accuracy = tf.metrics.accuracy(labels=tf.argmax(ys, 1), predictions=tf.argmax(output, 1))[1]
train = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

accuracy_his = []

for step in range(1001):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    sess.run([train], feed_dict={xs: b_x, ys: b_y})

    if step % 50 == 0:
        test_x, test_y = mnist.test.next_batch(BATCH_SIZE_TEST)
        l, op, ac = sess.run([loss, output, accuracy], feed_dict={xs: test_x, ys: test_y})
        accuracy_his.append(ac)
        # print(l, ac)
        # print(np.argmax(test_y, 1))
        # print(np.argmax(op, 1))

plt.plot(accuracy_his)
plt.ylim((0.5, 1))
plt.show()