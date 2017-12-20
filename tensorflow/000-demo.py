import numpy as np

x = np.random.rand(2, 2, 2)
# r, g, b = x[:,:,0], x[:,:,1], x[:,:,2]
# gray = 0.29*r + 0.58*g + 0.11*b
# gray = np.mean(x, -1)
# print(x)
# print(gray)
# print(x.flatten())
import tensorflow as tf

ty = tf.placeholder(tf.float32, [None, 10])
output = tf.layers.dense(10, 10)

accuracy = tf.metrics.accuracy(labels=tf.argmax(ty, 1), predictions=tf.argmax(output, 1))[1]

print(x)
print(np.argmax(x))









