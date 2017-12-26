import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

LR = 0.002
COMPARE_COLS = 5
BATCH_SIZE = 64

mnist = input_data.read_data_sets('./mnist', one_hot=False)

xs = tf.placeholder(tf.float32, [None, 28*28])

ACTI = tf.nn.tanh

e0 = tf.layers.dense(xs, 256, ACTI)
e1 = tf.layers.dense(e0, 64, ACTI)
e2 = tf.layers.dense(e1, 32, ACTI)
encoded = tf.layers.dense(e2, 2)

d0 = tf.layers.dense(encoded, 32, ACTI)
d1 = tf.layers.dense(d0, 64, ACTI)
d2 = tf.layers.dense(d1, 256, ACTI)
decoded = tf.layers.dense(d2, 28*28, tf.nn.sigmoid)

loss = tf.losses.mean_squared_error(labels=xs, predictions=decoded)
train = tf.train.AdamOptimizer(LR).minimize(loss)
accuracy = tf.metrics.accuracy(labels=tf.argmax(xs, 1), predictions=tf.argmax(decoded, 1))[1]

sess = tf.Session()
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
saver = tf.train.Saver()

view_data = mnist.test.images[:COMPARE_COLS]

f, a = plt.subplots(2, COMPARE_COLS, figsize=(5, 2))

for i in range(COMPARE_COLS):
    a[0][i].imshow(view_data[i].reshape(28, 28), cmap='rainbow')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

dir_path = './auto_encode_params/'
if os.path.isfile(dir_path):
    print('restore')
    saver.restore(sess, dir_path)

# plt.show()
for step in range(8000):
    x, _ = mnist.train.next_batch(BATCH_SIZE)
    l, ac, _, en, de = sess.run([loss, accuracy, train, encoded, decoded], feed_dict={xs: x})

    if step % 100 == 0:
        print(l, ac)

        if ac > 0.011:
            print('save')
            saver.save(sess, dir_path, write_meta_graph=False)

        decoded_data = sess.run(decoded, feed_dict={xs: view_data})
        for i in range(COMPARE_COLS):
            a[1][i].imshow(decoded_data[i].reshape(28, 28), cmap='rainbow')
            a[1][i].set_xticks(())
            a[1][i].set_yticks(())
        plt.pause(0.5)
# plt.show()

# visualize in 2D plot
test_x = mnist.test.images[:5000]
test_y = mnist.test.labels[:5000]

encoded_data = sess.run(encoded, {xs: test_x})
f = plt.figure(2, figsize=(10, 10))
ll = plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=test_y, cmap='rainbow', s=10)
plt.legend(loc='best')
plt.show()

# visualize in 3D plot
# test_x = mnist.test.images[:200]
# test_y = mnist.test.labels[:200]
#
# encoded_data = sess.run(encoded, {xs: test_x})
# fig = plt.figure(2)
# ax = Axes3D(fig)
# X, Y, Z = encoded_data[:, 0], encoded_data[:, 1], encoded_data[:, 2]
# for x, y, z, s in zip(X, Y, Z, test_y):
#     ax.text(x, y, z, s, backgroundcolor=cm.rainbow(int(255*s/9)))
#
# ax.set_xlim(X.min(), X.max())
# ax.set_ylim(Y.min(), Y.max())
# ax.set_zlim(Z.min(), Z.max())
# plt.show()
