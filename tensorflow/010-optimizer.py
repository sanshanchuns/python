import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 32

#fake data
x = np.linspace(-1, 1, 300)[:, np.newaxis]
y = np.power(x, 2) + np.random.normal(0, 0.05, x.shape)

class Net():
    def __init__(self, opt, **kwargs):
        self.x = tf.placeholder(tf.float32, [None, 1])
        self.y = tf.placeholder(tf.float32, [None, 1])
        hidden = tf.layers.dense(self.x, 20, tf.nn.relu)
        output = tf.layers.dense(hidden, 1)
        self.loss = tf.losses.mean_squared_error(output, self.y)
        self.train = opt(0.01, **kwargs).minimize(self.loss)


net_SGD = Net(tf.train.GradientDescentOptimizer)
net_Momentum = Net(tf.train.MomentumOptimizer, momentum=0.9)
net_RMSprop = Net(tf.train.RMSPropOptimizer)
net_Adam = Net(tf.train.AdamOptimizer)

nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]
losses_his = [[], [], [], []]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(300):
    index = np.random.randint(0, x.shape[0], BATCH_SIZE)
    batch_x, batch_y = x[index], y[index]

    for net, loss_his in zip(nets, losses_his):
        _, loss = sess.run([net.train, net.loss], feed_dict={net.x: batch_x, net.y: batch_y})
        loss_his.append(loss)

lables = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for label, loss_his in zip(lables, losses_his):
    plt.plot(loss_his, label=label)
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Losses')
plt.ylim((0, 0.02))
plt.show()