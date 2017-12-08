
import tensorflow as tf

m1 = tf.placeholder(tf.float32)
m2 = tf.placeholder(tf.float32)

# output = tf.matmul(m1, m2)  #如果是矩阵相乘，则feed的数据必须至少是2D
output = tf.multiply(m1, m2)  #m1, m2 shape 必须一样才可以相乘

with tf.Session() as sess:
    # value = sess.run(output, feed_dict={m1:[[1, 2, 3], [4, 5, 6]], m2:[[1, 4], [2, 5], [3, 6]]})
    value = sess.run(output, feed_dict={m1: 2, m2: 3})
    print(value)