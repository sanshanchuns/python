import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data

tf.set_random_seed(1)

BATCH_SIZE = 64
EPOCH = 8000
LR = 0.002
N_TEST_IMG = 5

mnist = input_data.read_data_sets('./mnist', one_hot=True)

xs = tf.placeholder(tf.float32, [None, 28*28])

l1 = tf.layers.dense(xs, 128, tf.nn.tanh)
l2 = tf.layers.dense(l1, 64, tf.nn.tanh)
l3 = tf.layers.dense(l2, 12, tf.nn.tanh)
encode = tf.layers.dense(l3, 3)

l3 = tf.layers.dense(encode, 12, tf.nn.tanh)
l4 = tf.layers.dense(l3, 64, tf.nn.tanh)
l5 = tf.layers.dense(l4, 128, tf.nn.tanh)
decode = tf.layers.dense(l5, 28*28, tf.nn.sigmoid)

loss = tf.losses.mean_squared_error(labels=xs, predictions=decode)
train = tf.train.AdamOptimizer(LR).minimize(loss)
accuracy = tf.metrics.accuracy(labels=tf.argmax(xs, 1), predictions=tf.argmax(decode, 1))[1]

sess = tf.Session()
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
view_data = mnist.test.images[:N_TEST_IMG] #前5个数据

# original data (first row) for viewing
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data[i], (28, 28)), cmap='rainbow')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

for step in range(EPOCH):
    b_x, _ = mnist.train.next_batch(BATCH_SIZE)
    _, l, ac, en, de = sess.run([train, loss, accuracy, encode, decode], feed_dict={xs: b_x})

    if step % 100 == 0:
        print(l, ac)
        decode_output = sess.run([decode], feed_dict={xs: view_data})
        print(decode_output)
        for i in range(N_TEST_IMG):
            a[1][i].clear()
            a[1][i].imshow(np.reshape(decode_output[i], (28, 28)), cmap='rainbow')
            a[1][i].set_xticks(())
            a[1][i].set_yticks(())
            plt.pause(0.01)
