from bs4 import BeautifulSoup
import requests
import time

'''
https://www.tripadvisor.cn/Attractions-g60763-Activities-oa30-New_York_City_New_York.html#FILTERED_LIST
https://www.tripadvisor.cn/Attractions-g60763-Activities-oa60-New_York_City_New_York.html#FILTERED_LIST
'''

urls = ['https://www.tripadvisor.cn/Attractions-g60763-Activities-oa{}-New_York_City_New_York.html#FILTERED_LIST'.format(str(i)) for i in range(30, 900, 30)]

cates = []

def crawlingData(url):
    with open('./output.txt', 'a') as text:
        wb_data = requests.get(url)
        soup = BeautifulSoup(wb_data.text, 'lxml')
        titles = soup.select('#ATTR_ENTRY_ > div.attraction_clarity_cell > div > div > div.listing_info > div.listing_title > a')
        imgs = soup.select('img[width=180]')
        cates = soup.select('div.p13n_reasoning_v2')

        for title, img, cate in zip(titles, imgs, cates):
            data = {
                'title' : title.get_text(),
                'img' : img.get('src'),
                'cate' : list(cate.stripped_strings),
            }