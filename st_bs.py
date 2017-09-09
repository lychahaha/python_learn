#coding=utf-8
import urllib2
from bs4 import BeautifulSoup

url='http://www.baidu.com'

response=urllib2.urlopen(url)

soup=BeautifulSoup(response.read(),'html.parser',from_encoding='utf-8')

for node in soup.find_all('a'):
	print node.name,node['href'],node.get_text()