#coding=utf-8
import urllib2
import re,time
from bs4 import BeautifulSoup

class WC:
	def __init__(self,url):
		self.unsearch=set([url])
		self.searched=set()
		self.urlbase='http://baike.baidu.com'
		self.count=0
	
	def begin(self):
		while len(self.unsearch)!=0 and self.count<100:
			url=self.unsearch.pop()
			self.searched.add(url)
			self.count+=1
			self.search(url)
			time.sleep(0.3)
		print self.count
		
	def search(self,url):
		#print 'open:',url
		try:
			response=urllib2.urlopen(url)
			self.analyse(url,response.read())
		except:
			print 'error'
		
	def analyse(self,url,text):
		soup=BeautifulSoup(text,'html.parser',from_encoding='utf-8')
		print soup.find('title').get_text()
		for node in soup.find_all('a',href=re.compile(r'^/view/')):
			url=self.urlbase+node['href']
			if url not in self.searched and url not in self.unsearch:
				self.unsearch.add(url)

url='http://baike.baidu.com/subview/10701645/18372706.htm?fromtitle=Lovelive&fromid=7288832&type=syn'

wc=WC(url)
wc.begin()