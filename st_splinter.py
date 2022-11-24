#jh:class-base,lazy-type
from splinter import Browser

#创建浏览器
br = Browser('chrome')
b = Browser(user_agent="Mozilla/5.0 (iPhone; U; CPU like Mac OS X; en)")

#访问
br.visit('https://www.baidu.com')
#后退前进
br.back()
br.forward()
#重载
br.reload()
#关闭浏览器
br.quit()

#静态属性
br.title
br.html
br.url

#填写表单
br.fill('wd', 'hehe') #wd是name
br.attach_file('file', '/tmp/a.jpg')
br.choose('some-radio', 'radio-value')
br.check('some-check')
br.uncheck('some-check')
br.select('year', '03') #rj是option的value
br.select_by_text('year', '2015')

br.type('type', slowly=True)???


#标签页
##查找
br.windows #所有标签页
br.windows[0] # 第一个标签页
br.windows.current #当前标签页
br.windows.current = br.windows[2] #设置当前标签页
##操作
win = br.windows[0]
win.is_current #是否是当前标签页
win.is_current = True #设置为当前标签页
win.next #下一个标签页
win.prev #上一个标签页
win.close() #关闭
win.close_others() #关闭其他所有标签页


#元素
##查找
br.find_by_css('h1')
br.find_by_xpath('//h1')
br.find_by_tag('h1')
br.find_by_name('name')
br.find_by_text('Hello World!')
br.find_by_id('firstheader')
br.find_by_value('query') #??
##获取
lists = br.find_by_tag('h1')
ele = lists.first
ele = lists.last
ele = lists[3]
##父亲儿子
ele.parent
##链接查找和点击
links = br.find_link_by_href('http://example.com')
links = br.find_link_by_partial_href('example')
links = br.find_link_by_text('Link for Example.com')
links = br.find_link_by_partial_text('for Example')
br.click_link_by_href('http://www.the_site.com/my_link')
br.click_link_by_partial_href('my_link')
br.click_link_by_text('my link')
br.click_link_by_partial_text('part of link text')
br.click_link_by_id('link_id')
##链式查找
div = br.find_by_tag('div')[0]
sub_div = div.find_by_name('ddd')
##操作
###鼠标操作
ele.mouse_over() #移到该元素上方
ele.mouse_out() #离开该元素
ele.click() #点击该元素
ele.double_click() #双击该元素
ele.right_click() #右击该元素
ele.scroll_to() #滚动到该元素
ele.drag_and_drop(ele2) #把ele拖放到ele2
ele.fill('xxx') #还有choose,select等表单操作
###属性
ele.visible #是否可操作
ele.tag_name
ele.text
ele.has_class('dd') #是否包含这个class

#driver元素
##与元素的关系
d_ele = ele._element
##查找
d_ele = br.driver.find_element_by_class_name('cc') #单个元素
d_eles = br.driver.find_elements_by_class_name('cc') #多个元素
br.driver.find_element_by_id('cc')
br.driver.find_element_by_name('cc')
br.driver.find_element_by_tag_name('cc')
br.driver.find_element_by_link_text('cc')
br.driver.find_element_by_partial_link_text('cc')
br.driver.find_element_by_css_selector('cc')
br.driver.find_element_by_xpath('cc')
##上传文件
d_ele.send_keys('C:\\haha.jpg')



#iframe(上下文)
with br.get_iframe('iframe_id') as iframe:
    div = br.find_by_id('xxx')







# xpath
//input #查找所有的input
//input[@class="as bs"] #带属性筛选（属性必须全部写出来）
//input[contains(@class,"as")] #包含某属性
//input[contains(@class,"as") or contains(@class,"as")] #支持与或运算
//input[@td] #包含该属性的属性筛选
//div/input #父亲是div的input
//div[@id="e"]//div #该div的所有div子孙
//div[@id="e"]/input[1] #该div的所有input儿子的第一个
//div[@id="e"]/input[last()] #该div的所有input儿子的最后一个
//div[@id="e"]/* #该div的所有儿子（*其实是通配符）
//div[@id="e"]/.. #该div的父亲
//div[@id="e"]/preceding-sibling::p[1] #该div的前一个哥哥p
//div[@id="e"]/following-sibling::p[1] #该div的后一个弟弟p
//div[@id="e"]/following-sibling::*[1] #该div的后一个弟弟


# css
div #全部div
div,p #全部div和p
div p #祖先是div的p
div>p #父亲是div的p
div+p #哥哥是div的p
div.cc #class为cc的所有div
.cc #class为cc的所有元素
div#cc #id为cc
div[class="as"] #带属性的筛选
div[class~="as"] #属性包含该值，而不需要完全相等
div[td] #包含该属性的筛选