
import tensorflow as tf
import numpy as np

m1 = tf.constant(3*np.ones([1, 2]))
m2 = tf.constant(2*np.ones([2, 1]))

product = tf.matmul(m1, m2)

print(tf.Session().run(product))

with tf.Session() as sess:
    print(sess.run(product))