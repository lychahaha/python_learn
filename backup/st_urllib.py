#jh:mix-base,lazy-type
import urllib.request

url = 'http://www.baidu.com'

#urlopen
resp = urllib.request.urlopen(url)#resp
resp = urllib.request.urlopen(url, timeout=2)
resp = urllib.request.urlopen(req)
#能自动处理302


#req
req = urllib.request.Request(url, headers={
	'Connection': 'Keep-Alive',
    'Accept': 'text/html, application/xhtml+xml, */*',
    'Accept-Language': 'en-US,en;q=0.8,zh-Hans-CN;q=0.5,zh-Hans;q=0.3',
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko'
	})


#resp
data = resp.read()#bytes-utf8
resp.geturl()
resp.getcode()#标志码,如200,302


#把参数的字典转成url字符串
url_val = urllib.parse.urlencode(d)
url = url + url_val

