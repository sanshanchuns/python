from bs4 import BeautifulSoup
import requests
import time

url = 'http://bj.58.com/pbdn/'

# wb_data = requests.get(url)
# soup = BeautifulSoup(wb_data.text, 'lxml')
# print(soup.select('.pricebiao'))

def get_item_info(url):

    wb_data = requests.get(url)
    soup = BeautifulSoup(wb_data.text, 'lxml')
    data = {
        'title': soup.title.text,
        'price': soup.select('.pricebiao'),
        'area': soup.select('.fl'),
        'date': None,
        'cate': None,
        'views': None,
    }
    print(data)

get_item_info(url)