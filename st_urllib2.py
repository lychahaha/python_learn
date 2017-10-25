#jh:proc-base,lazy-type
#coding=utf-8

import urllib2
import cookielib

url='http://www.baidu.com'

#first
#response=urllib2.urlopen(url)

#second
#request=urllib2.Request(url)
#request.add_header('user-agent','Mozilla/5.0')
#response2=urllib2.urlopen(request)

#third
cj=cookielib.CookieJar()
opener=urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))
urllib2.install_opener(opener)
response3=urllib2.urlopen(url)

print response3.getcode()
print len(response3.read())