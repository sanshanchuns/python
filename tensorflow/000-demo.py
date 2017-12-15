import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# fake data

x = np.linspace(-1, 1, 300)[:, np.newaxis]  # (300, 1)
y = np.power(x, 2) + np.random.normal(0, 0.05, x.shape) #真正的值

# plt.scatter(x, y)
# plt.show()

# placehodler

xs = tf.placeholder(tf.float32, x.shape)
ys = tf.placeholder(tf.float32, y.shape)

# 搭建layer

hidden = tf.layers.dense(xs, 10, tf.nn.relu)
output = tf.layers.dense(hidden, 1) #预测值

loss = tf.losses.mean_squared_error(output, ys)
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    _, l, op = sess.run([train, loss, output], feed_dict={xs: x, ys: y})
    print(l)
    if i % 50 == 0:
        plt.cla()
        plt.scatter(x, y, s=10, c=x, cmap='rainbow')
        plt.plot(x, op, 'r-', lw=1)
        plt.pause(0.1)
plt.show()









