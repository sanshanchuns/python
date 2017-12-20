from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import tensorflow as tf
from pathlib import Path

IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = 4
BATCH_SIZE = 20
LR = 0.001

# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

CHAR_SET = number + alphabet + ALPHABET + ['_']  # 如果验证码长度小于4, '_'用来补齐
CHAR_SET_LEN = len(CHAR_SET)


def random_captcha_text(char_set=CHAR_SET, char_size=MAX_CAPTCHA):
    captcha_text = []
    for i in range(char_size):
        captcha_text.append(random.choice(char_set))
    return captcha_text


def gen_captcha_text_and_image(number_only=False):
    char_set = CHAR_SET
    if number_only:
        char_set = number
    captcha_text = random_captcha_text(char_set)
    captcha_text = ''.join(captcha_text)

    captcha_image = ImageCaptcha().generate_image(captcha_text) #实例方法
    captcha_image = np.array(captcha_image)

    return captcha_text, captcha_image


# text, image = gen_captcha_text_and_image()
# 图像大小

# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2: #如果是2D以上, 降D
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img

"""
cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。
np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))  # 在图像上补2行，下补3行，左补2行，右补2行
"""

# 文本转向量
#向量的长度是 4 * (10 + 26 + 26 + 1) = 252
#[0-9], [48, 57]
#[A-Z], [65, 90]
#[a-z], [97, 122]


def text2vec(txt):
    text_len = len(txt)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)  #(252,)

    def char2pos(c):
        if c == '_':
            k = 62  #位置是62
            return k
        k = ord(c) - 48  #48是0，最开始的offset
        if k > 9:  #不是数字
            k = ord(c) - 55 #大小写字母
            if k > 35: #不是大写写字母，因为大写的Z是90 - 55 = 35
                k = ord(c) - 61  # a(97)
                if k > 61: #超出小子字母z(122)的范围
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(txt):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector


# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    txt = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        txt.append(chr(char_code))
    return "".join(txt)


def index2text(index):
    txt = []
    for i, idx in enumerate(index):
        txt.append(CHAR_SET[idx])
    return ''.join(txt)

"""
#向量（大小MAX_CAPTCHA*CHAR_SET_LEN）用0,1编码 每63个编码一个字符，这样顺利有，字符也有
vec = text2vec("F5Sd")
text = vec2text(vec)
print(text)  # F5Sd
vec = text2vec("SFd5")
text = vec2text(vec)
print(text)  # SFd5
"""


# 生成一个训练batch
def next_batch(batch_size=20, number_only=False):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    # 有时生成图像大小不是(60, 160, 3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image(number_only)
            if image.shape == (60, 160, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)

        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y

xs = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
xs_4d = tf.reshape(xs, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
ys = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])


def crack_captcha_cnn():
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


def train_crack_captcha_cnn():
    output = crack_captcha_cnn()

    #ys (20, 252) -> (1, 4, 63)   output (20, 252)
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=output))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ys, logits=output))
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
            b_x, b_y = next_batch(BATCH_SIZE)
            sess.run([train], feed_dict={xs: b_x, ys: b_y})
            if i % 100 == 0:
                t_x, t_y = next_batch(BATCH_SIZE)
                l, op, ac = sess.run([loss, output, accuracy], feed_dict={xs: t_x, ys: t_y})
                print(i, l, ac)
                if l < 0.01:
                    print('resave')
                    saver.save(sess, './cnn_params/', write_meta_graph=False)  # meta_graph is not recommended
                index = np.argmax(op[0, :].reshape(4, 63), 1)
                print(index2text(index))
                print(vec2text(t_y[0, :]))

    else:

        for i in range(10000):
            b_x, b_y = next_batch(BATCH_SIZE)
            sess.run([train], feed_dict={xs: b_x, ys: b_y})
            if i % 100 == 0:
                t_x, t_y = next_batch(BATCH_SIZE, number_only=True)
                l, op, ac = sess.run([loss, output, accuracy], feed_dict={xs: t_x, ys: t_y})
                print(i, l, ac)
                if l < 0.03:
                    print('save')
                    saver.save(sess, './cnn_params/', write_meta_graph=False)  # meta_graph is not recommended
                index = np.argmax(op[0, :].reshape(4, 63), 1)
                print(index2text(index))
                print(vec2text(t_y[0, :]))


if __name__ == '__main__':

    # text, image = gen_captcha_text_and_image()
    # image = convert2gray(image)
    # print(len(image.shape))

    # f = plt.figure()
    # ax = f.add_subplot(111)
    # ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    # plt.imshow(image)
    # plt.show()

    train_crack_captcha_cnn()

