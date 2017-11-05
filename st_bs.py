#jh:class-base,key-type
import urllib
import urllib.request
from bs4 import BeautifulSoup
import re

##BeautifulSoup
url='http://www.baidu.com'
resp = urllib.request.urlopen(url)
data = resp.read()
soup = BeautifulSoup(data, 'html.parser')



##Tag
soup.title
soup.head
soup.a #有多个会返回第一个

#标签名字
a.name #返回'a'

#属性
a.attrs #dict,标签里的属性
a.attrs['href'] #可读可写
#下面的也行
a['href']
a.get('href')

#文字
p.text

#儿子
div.contents #list,儿子列表,儿子有可能是Tag或NavigableString

#搜索
soup.find_all('a')
soup.find_all('div', class_='bri')
soup.find_all('div', attrs={'class':'bri','name':'tj_briicon'})
soup.find_all(func) #func:tag->bool
soup.find_all(['div','p'])
soup.find_all(re.compile('^b'))
div.find_all('a') #在div中搜索


##NavigableString
p.string #可能是None?



##Comment
a.string #注释的时候?

