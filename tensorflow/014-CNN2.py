import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import numpy as np
import pkg_dataset.dataset_captcha as dataset

IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = 4
BATCH_SIZE = 20
LR = 0.001
CHAR_SET_LEN = 63  #(10 + 26 + 26 + 1)

xs = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
xs_4d = tf.reshape(xs, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
ys = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])


def create_cnn_layer():
    c1 = tf.layers.conv2d(xs_4d, 16, 5, 1, 'same', activation=tf.nn.relu)  # -> 60*160*16
    c1 = tf.layers.dropout(c1, rate=0.5)
    p1 = tf.layers.max_pooling2d(c1, 2, 2)  # -> 30*80*16
    c2 = tf.layers.conv2d(p1, 32, 5, 1, 'same', activation=tf.nn.relu)  # -> 30*80*32
    c2 = tf.layers.dropout(c2, rate=0.5)
    p2 = tf.layers.max_pooling2d(c2, 2, 2)  # -> 15*40*32
    c3 = tf.layers.conv2d(p2, 64, 5, 1, 'same', activation=tf.nn.relu)  # -> 15*40*64
    c3 = tf.layers.dropout(c3, rate=0.5)
    p3 = tf.layers.max_pooling2d(c3, 2, 2) # -> 15*40*64
    output = tf.layers.dense(tf.reshape(p3, [-1, 7 * 20 * 64]), MAX_CAPTCHA * CHAR_SET_LEN)
    return output


def train_cnn():
    output = create_cnn_layer()

    #ys (20, 252) -> (1, 4, 63)   output (20, 252)
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=ys, logits=output)
    # 这里的loss函数可以直接用losses集合中的函数，也可以用nn里的函数，差别就是要多加一层 reduce_mean
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ys, logits=output))
    accuracy = tf.metrics.accuracy(labels=tf.argmax(ys, 1), predictions=tf.argmax(output, 1))[1]
    train = tf.train.AdamOptimizer(LR).minimize(loss)

    sess = tf.Session()
    saver = tf.train.Saver()  # define a saver for saving and restoring
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    file_path = './cnn_params/'
    if Path(file_path).exists():
        saver.restore(sess, file_path)
        print('restore')

        for i in range(10000):
            b_x, b_y = dataset.next_batch(BATCH_SIZE)
            sess.run([train], feed_dict={xs: b_x, ys: b_y})
            if i % 100 == 0:
                t_x, t_y = dataset.next_batch(BATCH_SIZE)
                l, op, ac = sess.run([loss, output, accuracy], feed_dict={xs: t_x, ys: t_y})
                print(i, l, ac)
                if l < 0.01:
                    print('resave')
                    saver.save(sess, file_path, write_meta_graph=False)  # meta_graph is not recommended
                op_index = np.argmax(op[0, :].reshape(4, 63), 1)
                print(dataset.index2text(op_index))
                print(dataset.vec2text(t_y[0, :]))

    else:

        for i in range(10000):
            b_x, b_y = dataset.next_batch(BATCH_SIZE)
            sess.run([train], feed_dict={xs: b_x, ys: b_y})
            if i % 100 == 0:
                t_x, t_y = dataset.next_batch(BATCH_SIZE, number_only=True)
                l, op, ac = sess.run([loss, output, accuracy], feed_dict={xs: t_x, ys: t_y})
                print(i, l, ac)
                if l < 0.03:
                    print('save')
                    saver.save(sess, file_path, write_meta_graph=False)  # meta_graph is not recommended
                op_index = np.argmax(op[0, :].reshape(4, 63), 1)
                print(dataset.index2text(op_index))
                print(dataset.vec2text(t_y[0, :]))


if __name__ == '__main__':

    train_cnn()

