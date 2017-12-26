import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt

LR = 0.002
COMPARE_COLS = 5
BATCH_SIZE = 64

mnist = input_data.read_data_sets('./mnist', one_hot=True)

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

test_images = mnist.test.images[:COMPARE_COLS]
test_lables = mnist.test.labels[:COMPARE_COLS]

f, a = plt.subplots(2, COMPARE_COLS, figsize=(5, 2))

for i in range(COMPARE_COLS):
    a[0][i].imshow(test_images[i].reshape(28, 28), cmap='rainbow')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

# plt.show()
for step in range(1000):
    x, _ = mnist.train.next_batch(BATCH_SIZE)
    l, ac, _, en, de = sess.run([loss, accuracy, train, encoded, decoded], feed_dict={xs: x})

    if step % 100 == 0:
        print(l, ac)

        output = sess.run(decoded, feed_dict={xs: test_images})
        for i in range(COMPARE_COLS):
            a[1][i].imshow(output[i].reshape(28, 28), cmap='rainbow')
            a[1][i].set_xticks(())
            a[1][i].set_yticks(())
        plt.pause(0.5)
# plt.show()

encoded_result = sess.run(encoded, feed_dict={xs: test_images})
print(encoded_result.shape)
plt.scatter(encoded_result[:, 0], encoded_result[:, 1], c=test_lables)
plt.colorbar()
plt.show()