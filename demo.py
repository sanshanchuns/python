
import tensorflow as tf
import numpy as np

#fake data
# x = np.random.rand(100).astype(np.float32)
# y_label = x*0.1 + 0.3
#
# # create tensorflow structure
#
# w = tf.Variable(tf.random_uniform([1], -1, 1))
# b = tf.Variable(tf.zeros([1]))
#
# y = w*x + b
#
# loss = tf.reduce_mean(tf.square(y - y_label))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)
#
# init = tf.global_variables_initializer()
#
# # create tensorflow structure
#
# sess = tf.Session()
# sess.run(init)  # start from init
#
# for step in range(201):
#     sess.run(train) # start train
#     if step % 20 == 0:
#         print(step, sess.run(w), sess.run(b))

# x = tf.constant(1)
# y = tf.constant(2)
#
# print(tf.Session().run(tf.add(x, y)))
# tf.reduce_sum(x)  # 6
# tf.reduce_sum(x, 0)  # [2, 2, 2]
# tf.reduce_sum(x, 1)  # [3, 3]
# tf.reduce_sum(x, 1, keep_dims=True)  # [[3], [3]]
# tf.reduce_sum(x, [0, 1])  # 6

# print(tf.Session().run(tf.reduce_sum(x, reduction_indices=[1])))
# print(tf.Session().run(tf.reduce_sum(x, 1)))

x = np.random.normal(np.ones([2, 3, 4]))
print(x.shape) #有多少维度
print(x.size) #一共有多少个元素