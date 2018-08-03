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

import urllib
import chardet

uris = ['5ybmfAEnxqM', 'iQK2zXTiZmc', 'nb8AKLQtUrY', 'R1PXQB-2eU0', 'UM_5b3C4Jtw', 'WiyCqO82_-A']

urls = ['https://s3.cn-north-1.amazonaws.com.cn/u-subs-vtt/zh-cn/{}.vtt'.format(i) for i in uris]

video_urls = ['https://s3.cn-north-1.amazonaws.com.cn/u-vid-hd/{}.mp4'.format(i) for i in uris]

target_path = '/Users/leo/Downloads/'

# for index, url_tuple in enumerate(zip(urls, video_urls)):
    # print index, url_tuple[0], url_tuple[1]
    # urllib.urlretrieve(url_tuple[0], target_path + uris[index] + '.srt')
    # urllib.urlretrieve(url_tuple[1], target_path + uris[index] + '.mp4')

for path in [target_path + uri + '.srt' for uri in uris]:
    with open(path, 'r') as text:
        # raw_text = text.read().decode(encoding='UTF-8',errors='strict').split()
        raw_text = text.read().split()
        



























