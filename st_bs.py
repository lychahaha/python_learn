#coding=utf-8
import urllib
import urllib.request
from bs4 import BeautifulSoup

url='http://www.baidu.com'

response=urllib.request.urlopen(url)

soup=BeautifulSoup(response.read(), 'html.parser')

for node in soup.findAll('a'):
	print node.name,node['href'],node.get_text()