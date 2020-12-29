from selenium import webdriver

# 创建浏览器
browser = webdriver.Chrome()
# 打开页面
browser.get('https://www.baidu.com')
# 关闭浏览器
browser.close()

# 查找元素
div = browser.find_element_by_tag_name('div')
div = browser.find_element_by_class_name('abc')
div = browser.find_element_by_id('abc')
div = browser.find_element_by_name('abc')
# 查找多个元素
divs = browser.find_elements_by_tag_name('div')

# 交互
btn.click()
input_.send_keys('my username')
input_.clear()

# 属性获取
div.get_attribute('class')
div.text
div.id
div.tag_name
div.size
div.location

# 执行js脚本
browser.execute_script('arguments[0].scrollIntoView();', div)
