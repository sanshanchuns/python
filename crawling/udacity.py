#!/Users/leo/.pyenv/shims/python
# -*- coding: utf-8 -*-

# from bs4 import BeautifulSoup
# import requests
# import time

# '''
# https://www.tripadvisor.cn/Attractions-g60763-Activities-oa30-New_York_City_New_York.html#FILTERED_LIST
# https://www.tripadvisor.cn/Attractions-g60763-Activities-oa60-New_York_City_New_York.html#FILTERED_LIST
# '''

# urls = ['https://classroom.udacity.com/nanodegrees/nd112/parts/503cfd7e-28c5-47c4-a9ec-e6cc59b4ba30']

# cates = []

# def crawlingData(url):
#     with open('./output.txt', 'a') as text:
#         wb_data = requests.get(url)
#         soup = BeautifulSoup(wb_data.text, 'lxml')
#         print(soup)

#         # for title, img, cate in zip(titles, imgs, cates):
#         #     data = {
#         #         'title' : title.get_text(),
#         #         'img' : img.get('src'),
#         #         'cate' : list(cate.stripped_strings),
#         #     }

import os
import urllib
import chardet
from os import listdir
from os.path import isfile, join
import re
import sys 
reload(sys) # Python2.5 初始化后会删除 sys.setdefaultencoding 这个方法，我们需要重新载入 
sys.setdefaultencoding('utf-8')

# uris = ['5ybmfAEnxqM', 'iQK2zXTiZmc', 'nb8AKLQtUrY', 'R1PXQB-2eU0', 'UM_5b3C4Jtw', 'WiyCqO82_-A']
# uris = ['iQK2zXTiZmc']

def enumerateDirectory(path):
    file_paths_without_extension = []
    file_names = []
    paths = os.walk(target_path)
    for path, dir_list, file_list in paths:
        for file_name in file_list:
            if file_name.endswith('.mp4'):
                file_path = os.path.join(path, os.path.splitext(file_name)[0])
                file_paths_without_extension.append(file_path)
                file_names.append(file_name)
                print file_path

def downloadSrcAndParse(target_path):
    file_paths_without_extension = []
    file_names = []
    paths = os.walk(target_path)
    for path, dir_list, file_list in paths:
        for file_name in file_list:
            if file_name.endswith('.mp4'):
                file_path = os.path.join(path, os.path.splitext(file_name)[0])
                file_paths_without_extension.append(file_path)
                file_names.append(file_name)
                # print file_path

    # file_names = [os.path.splitext(f)[0] for f in listdir(target_path) if os.path.splitext(f)[1].startswith('.mp4')]
    urls = ['https://s3.cn-north-1.amazonaws.com.cn/u-subs-vtt/zh-cn/{}.vtt'.format(i.split('.')[1]) for i in file_names]

    en_urls = ['https://s3.cn-north-1.amazonaws.com.cn/u-subs-vtt/en-us/{}.vtt'.format(i.split('.')[1]) for i in file_names]

    video_urls = ['https://s3.cn-north-1.amazonaws.com.cn/u-vid-hd/{}.mp4'.format(i.split('.')[1]) for i in file_names]

    for index, url_tuple in enumerate(zip(urls, en_urls)):
        # print index, url_tuple[0], url_tuple[1]
        src_path = file_paths_without_extension[index] + '.srt'
        src_en_path = file_paths_without_extension[index] + '.en.srt'
        # if not os.path.exists(src_path):
        #     urllib.urlretrieve(url_tuple[0], src_path)
        if not os.path.exists(src_en_path):
            urllib.urlretrieve(url_tuple[1], src_en_path)

    # for path in [uri + '.srt' for uri in file_paths_without_extension]:
    #     with open(path, 'r') as text:

    #         raw_text = text.read()
    #         if len(raw_text) == 0:
    #             print path + ' 长度为0'
    #             continue

    #         encode_result = chardet.detect(raw_text)['encoding']
    #         # print path + ' ' + encode_result

    #         if not raw_text.startswith('<?xml'):
    #             output = re.sub('<.*?>', '', raw_text)

    #             with open(path, 'w') as output_file:
    #                 if isinstance(output, str):
    #                     print path + ' str'
    #                     output = output.decode(encode_result).replace(u'\ufeff', '')
    #                     output_file.write(output.encode('gb2312', 'ignore'))
    #                 elif isinstance(output, unicode):
    #                     print path + ' unicode'
    #                     output = output.replace(u'\ufeff', '')
    #                     output_file.write(output.encode('gb2312', 'ignore'))

    for path in [uri + '.en.srt' for uri in file_paths_without_extension]:
        # os.remove(path)
        with open(path, 'r') as text:
            raw_lines = text.read().split('\n')
            lines = [line.replace('<v English>', '').replace('</v>', '') for line in raw_lines if not line.startswith('<?xml')]
            output = '\n'.join(lines)
            with open(path, 'w') as output_file:
                output_file.write(output)


if __name__ == '__main__':
    # target_path = '/Volumes/SAMSUNG64G/Digital Marketing/'
    target_path = '/Users/leozhu/Movies/DM/'
    downloadSrcAndParse(target_path)
    # enumerateDirectory(target_path)

        



























for index, url_tuple in enumerate(zip(urls, video_urls)):
    # print index, url_tuple[0], url_tuple[1]
    urllib.urlretrieve(url, target_path + uris[index] + '.vtt')
    urllib.urlretrieve(url, target_path + uris[index] + '.mp4')
