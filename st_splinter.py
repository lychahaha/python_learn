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

#填写表单
br.fill('wd', 'hehe') #wd是name
br.attach_file('file', '/tmp/a.jpg')
br.choose('some-radio', 'radio-value')
br.check('some-check')
br.uncheck('some-check')
br.select('year', '03') #rj是option的value
br.select_by_text('year', '2015')

br.type('type', slowly=True)???


#静态属性
br.title
br.html
br.url



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
divs = br.find_by_tag('div')
div = divs.first.find_by_name('ddd')
##操作
###鼠标操作
ele.mouse_over() #移到该元素上方
ele.mouse_out() #离开该元素
ele.click() #点击该元素
ele.double_click() #双击该元素
ele.right_click() #右击该元素
ele.drag_and_drop(ele2) #把ele拖放到ele2
ele.fill('xxx') #还有choose,select等表单操作
###属性
ele.visible #是否可操作
ele.tag_name
ele.text
