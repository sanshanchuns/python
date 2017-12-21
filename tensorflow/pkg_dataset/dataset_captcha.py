from captcha.image import ImageCaptcha
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt

IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = 4
BATCH_SIZE = 20

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


if __name__ == '__main__':

    text, image = gen_captcha_text_and_image()

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)
    plt.show()