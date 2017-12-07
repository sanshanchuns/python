
import tensorflow as tf
import numpy as np

#fake data
x = np.random.rand(100).astype(np.float32)
y_label = x*0.1 + 0.3

# create tensorflow structure

w = tf.Variable(tf.random_uniform([1], -1, 1))
b = tf.Variable(tf.zeros([1]))

y = w*x + b

loss = tf.reduce_mean(tf.square(y - y_label))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# create tensorflow structure

sess = tf.Session()
sess.run(init)  # start from init

for step in range(201):
    sess.run(train) # start train
    if step % 20 == 0:
        print(step, sess.run(w), sess.run(b))