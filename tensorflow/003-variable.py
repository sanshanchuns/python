
import tensorflow as tf

state = tf.Variable(1, name='constant')
















# state = tf.Variable(1, name='counter')
# one = tf.constant(1)
#
# update = tf.assign(state, tf.add(state, one))
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#
#     for i in range(3):
#         sess.run(update)
#
#         print(sess.run(state))



#
# with tf.Session() as sess:
#     sess.run(init) #同一个session激活所有变量， 变量激活之前无法使用, 常量不需要激活就可以使用
#
#     for i in range(3):
#         sess.run(update)
#
#         print(sess.run(state))
