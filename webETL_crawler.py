#coding: utf-8
import requests
import re
from bs4 import BeautifulSoup

# craw html from url(list) and return web text with a string(list)
# the result text is divided with ".", without handling more punctuations like "，", "。", "[]"
def url_to_text(url,mode='direct') -> str:
    '''
    mode : diret, webETL
    '''

    # use API
    if mode == 'direct':
        web = requests.get(url)
        web_list = [web.text]
    elif mode == 'webETL':
        # the API only use these parameters
        JSON = {
            "urls": url,
            "cache": False,
            "timeout": 15000
        }
        web = requests.post("http://140.115.54.45:6789/post/crawler/static/html", json=JSON)
        web_list = web.json()
    # filter the content of return string(list)
    return_list = []
    for element in web_list:
        soup = BeautifulSoup(element, 'html.parser')
        temp = soup.get_text()
        dr = re.compile(r'(\t)+(\n)*|(\t)*(\n)+|。+')
        temp = dr.sub('.',temp)
        dr = re.compile('[.]+')
        temp = dr.sub('.',temp)
        dr = re.compile('\u3000+|\xa0+|\\+')
        temp = dr.sub('',temp)
        return_list.append(temp)

    return ''.join(return_list)

if __name__ == '__main__':
    txts = url_to_text('https://tw.dictionary.search.yahoo.com/search?p=crawler&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8&guce_referrer_sig=AQAAAAOzJd1cWrW7brHgF-nj5x50Ooadq_OIuL4NJfKt0MJoROGtNKq6kQ4iQy67yalXir2tW5jivSAPqSkVzaIxZzB0Vs865UcUBdwstnPWy8mBuvNSMhav9sF7fEQLHWSuR16bDPOn7eXbk46d-IoPGjDL1BsqNZyo4iDW27Wn0xY0',mode='direct')
    print(txts)