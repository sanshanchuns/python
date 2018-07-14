import numpy as np
import matplotlib.pyplot as plt
import string
import random
import os

if not os.path.exists('./craling/output.txt'):
    with open('./craling/Walden.txt', 'r') as text:
        words = [raw_word.strip(string.punctuation).lower() for raw_word in text.read().split()]
        word_index = set(words)
        word_list_filter = {}
        for word in sorted(word_index):
            if (words.count(word) > 500):
                word_list_filter[words.count(word)] = word
        with open('./craling/output.txt', 'w') as output:
            output.write(str(word_list_filter))

    # print(word_list_filter)
with open('./craling/output.txt') as text:
    word_list_filter = eval(text.read())
    count_x = sorted(list(word_list_filter.keys()), reverse=True)  # count
    word_y = list(word_list_filter.values())  # words
    plt.bar(range(len(count_x)), count_x)
    plt.xticks(range(len(word_y)), [word_list_filter[x] for x in count_x])
    plt.show()

# print(len(string.ascii_letters))

# test = {key:value for key,value in zip([string.ascii_letters[i] for i in range(26)], [random.randint(0, 100) for j in range(26)])}
# print(test)
# print(list(test.keys()))
# print(list(test.values()))
# x = list(test.keys())
# y = list(test.values())

# print(x)
# print(y)

# plt.bar(range(26), y)
# plt.show()

# plt.bar(range(len(y)),y)
# plt.xticks(range(len(x)), x)
# plt.show()

# import matplotlib.pyplot as plt
# import math
# import numpy as np

# alphabet = range(0, 26)
# # secondLine = [letter + 97 for letter in alphabet]
# secondLine = [chr(97 + x) for x in range(26)]

# plt.plot(alphabet, secondLine)
# plt.xticks(range(26), secondLine)
# plt.pause(5)

    #     print('{} - {} times'.format(word, words.count(word)))

    # plt.plot(sorted(word_index), [words.count(word) for word in sorted(word_index)])
    # plt.show()



    # word_list = [('{} - {} times'.format(word, words.count(word))) for word in sorted(word_index)]
    # print(word_list)

# words = [str(t).lower() for t in l]
# word_index = set(words)
# for word in sorted(word_index):
#     print('{} - {} times'.format(word, words.count(word)))

# with open('./demo.py', 'r') as text:
#     print(text.read().split())

# path = './Walden.txt'

# with open(path, 'r') as text:
#     words = text.read().split()
#     words = [raw_word.strip(string.punctuation).lower() for raw_word in text.read().split()]
#     words_index = set(words)
#     counts_dict = {index:words.count(index) for index in words_index}

# for word in sorted(counts_dict, key=lambda x: counts_dict[x], reverse=True):
#     print('{} -- {} times'.format(word, counts_dict[word]))

    # print(words)
    # for word in words:
    #     print('{}-{} times'.format(word, words.count(word)))