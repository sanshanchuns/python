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

uris = ['5ybmfAEnxqM', 'nb8AKLQtUrY']

urls = ['https://s3.cn-north-1.amazonaws.com.cn/u-subs-vtt/zh-cn/{}.vtt'.format(i) for i in uris]

video_urls = ['https://s3.cn-north-1.amazonaws.com.cn/u-vid-hd/{}.mp4'.format(i) for i in uris]

target_path = '/Users/leo/Downloads/'

for index, url_tuple in enumerate(zip(urls, video_urls)):
    # print index, url_tuple[0], url_tuple[1]
    urllib.urlretrieve(url, target_path + uris[index] + '.vtt')
    urllib.urlretrieve(url, target_path + uris[index] + '.mp4')
