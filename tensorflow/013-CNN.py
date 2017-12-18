import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

BATCH_SIZE = 50
BATCH_SIZE_TEST = 10
LR = 0.01

tf.set_random_seed(1)
np.random.seed(1)

mnist = input_data.read_data_sets('mnist', one_hot=True)

test_x = mnist.test.images[:BATCH_SIZE_TEST]
test_y = mnist.test.labels[:BATCH_SIZE_TEST]

xs = tf.placeholder(tf.float32, [None, 784]) # 28*28
image = tf.reshape(xs, [-1, 28, 28, 1])
ys = tf.placeholder(tf.float32, [None, 10])

con1 = tf.layers.conv2d(image, 16, 5, 1, 'same', activation=tf.nn.relu) # 28*28*16
p1 = tf.layers.max_pooling2d(con1, 2, 2) # 14*14*16
con2 = tf.layers.conv2d(p1, 32, 5, 1, 'same', activation=tf.nn.relu) #14*14*32
p2 = tf.layers.max_pooling2d(con2, 2, 2) #7*7*32
output = tf.layers.dense(tf.reshape(p2, [-1, 7*7*32]), 10) # => (7*7*32, )

loss = tf.losses.softmax_cross_entropy(onehot_labels=ys, logits=output)
train = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for step in range(1001):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    sess.run([train], feed_dict={xs: b_x, ys: b_y})

    if step % 50 == 0:
        l, op = sess.run([loss, output], feed_dict={xs: test_x, ys: test_y})
        print(np.argmax(test_y, 1))
        print(l, np.argmax(op, 1))
