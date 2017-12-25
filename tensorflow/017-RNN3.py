import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

TIME_STEP = 10
INPUT_SIZE = 1
CELL_SIZE = 32
LR = 0.01

xs = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])
# xs_3d = tf.reshape(xs, [-1, TIME_STEP, INPUT_SIZE])
ys = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])

rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=CELL_SIZE)
init_s = rnn_cell.zero_state(batch_size=INPUT_SIZE, dtype=tf.float32)
outputs, final_s = tf.nn.dynamic_rnn(
    cell=rnn_cell,
    inputs=xs,
    initial_state=init_s,
    time_major=False,
)

# outputs # shape(batch, time_step, cell_size)
# output # shape(batch, time_step, input_size)

op_3d_2_2d = tf.reshape(outputs, [-1, CELL_SIZE])#3D->2D, 也可以
op_2d = tf.layers.dense(op_3d_2_2d, INPUT_SIZE) #全连接层是线性变换，因此2D数据是最好的
output = tf.reshape(op_2d, [-1, TIME_STEP, INPUT_SIZE])

loss = tf.losses.mean_squared_error(labels=ys, predictions=output)
train = tf.train.AdamOptimizer(LR).minimize(loss)
accuracy = tf.metrics.accuracy(labels=tf.argmax(ys, 1), predictions=tf.argmax(output, 1))[1]

sess = tf.Session()
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

plt.figure(1, figsize=(12, 5))

for step in range(60):
    start, end = step*np.pi, (step+1)*np.pi
    steps = np.linspace(start, end, TIME_STEP)
    x = np.sin(steps)[np.newaxis, :, np.newaxis]
    y = np.cos(steps)[np.newaxis, :, np.newaxis]

    if 'fs' not in globals():
        feed_dict = {xs: x, ys: y}
    else:
        feed_dict = {xs: x, ys: y, init_s: fs}

    _, l, ac, op, fs = sess.run([train, loss, accuracy, output, final_s], feed_dict=feed_dict)
    print(l, ac)

    plt.plot(steps, y.flatten(), 'r-')
    plt.plot(steps, op.flatten(), 'b--')
    plt.ylim((-1.2, 1.2))
    plt.pause(0.05)

plt.show()
