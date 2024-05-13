import requests

# get
resp = requests.get('https://baidu.com')

# post
resp = requests.post('https://baidu.com', \
                     {'aa':'bb'}, \
                     headers={'Content-Type':'application/json'})

resp.status_code #int,如200
resp.text #str,resp的正文
resp.request #查看对应的req
resp.url #url




import http.client

conn = http.client.HTTPConnection('12.34.56.78', '1234')
conn.request('POST', '/ab/c', data, headers)
resp = conn.getresponse()

resp.getcode()
resp.read().decode('utf-8')
resp.geturl()