from selenium import webdriver
from selenium.webdriver.support.ui import Select

# 创建浏览器
br = webdriver.Chrome()
# 浏览器基本操作
br.get('https://www.baidu.com') #打开页面
br.back() #后退
br.forward() #前进
br.close() #关闭
br.maximize_window() #最大化
br.minimize_window() #最小化
br.quit() #退出浏览器

# 窗口
br.window_handles #所有窗口
br.current_window_handle #当前窗口
br.switch_to.window(win) #切换窗口
win.close() #关闭窗口

# 浏览器属性
br.title
br.page_source
br.current_url

# cookie
br.add_cookie({'a':'b'})
br.get_cookie('a') #给的参数是cookie的name字段
br.get_cookies()

# iframe
br.switch_to.frame(iframe)
br.switch_to.frame('iframe_id')
br.switch_to.default_content() #切回最大的frame


# 查找元素(8种方式)
## 属性方式
div = br.find_element('id', 'kw') #根据id找
div = br.find_element('class name', 'kw') #根据class找
div = br.find_element('tag name', 'kw') #根据tag找
div = br.find_element('name', 'kw') #根据name找
## xpath和css
div = br.find_element('xpath', 'kw') #根据xpath找
div = br.find_element('css selector', 'kw') #根据css找
## link相关
div = br.find_element('link text', 'kw') #根据link text找
div = br.find_element('partial link text', 'kw') #根据部分link text找
# 查找多个元素
divs = br.find_elements('class name', 'kw') #找多个

# 交互
ele.click() #点击按钮
ele.send_keys('my username') #输入框填写
ele.clear() #清空输入框
ele.send_keys('D:/a.txt') #传输文件
br.execute_script("window.scrollTo(0,document.body.scrollHeight);") #滚动到底部
br.execute_script("arguments[0].scrollIntoView()", ele) #滚动到该元素和窗口顶部对齐
## select输入框
Select(ele).select_by_index(3) #从0开始
Select(ele).select_by_value('xx')
Select(ele).select_by_visible_text('xx')



# 元素属性
div.get_attribute('class')
div.get_attribute('outerHTML')
div.get_dom_attribute('xxx') #HTML代码固有属性
div.get_property('xxx') #js属性
div.get_attribute('xxx') #先看property，没才找dom_attribute
div.text
div.id
div.tag_name
div.size
div.location

# 执行js脚本
br.execute_script('arguments[0].scrollIntoView();', div)

# MHTML数据
from email.parser import Parser #用于解析MHT文件
import base64 

res = br.execute_cdp_cmd('Page.captureSnapshot', {}) #获取MHT数据(一个dict,只有一个key为"data",value是字符串)
open("a.mht", 'w', encoding='utf-8').write(res['data']) #保存为MHT文件
mht = Parser().parsestr(res['data']) #解析MHT

mht.is_multipart() #是否多part
for part in mht.walk(): #遍历每个part
    pass

part.get_content_type() #该part的type，如text/html、image/jpeg
part.get_content_maintype() #该part的type的前半段
part.get_content_subtype() #该part的type的后半段
part.keys() #part本身作为字典存储了一些键值对
part['Content-Location'] #该part的url
part.get_payload() #该part的文件内容
open("a.jpg",'wb').write(base64.b64decode(part.get_payload())) #part为图片时需b64解码
