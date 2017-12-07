
import tensorflow as tf
import numpy as np

x = np.random.sample(100).astype(np.float32)
y_label = 0.1*x + 0.3

# create linear
w = tf.Variable(tf.random_uniform([1], -1, 1))
b = tf.Variable(tf.zeros([1]))
y = w*x + b

# create structure
loss = tf.reduce_mean(tf.square(y - y_label))
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# start graph
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) #激活变量

# start training

for step in range(201):
    sess.run(train)

    if step % 20 == 0:
        print(step, sess.run(w), sess.run(b))

