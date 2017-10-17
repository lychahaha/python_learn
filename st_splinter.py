from splinter import Browser

browser = Browser('chrome')
browser.visit('http://www.baidu.com')
browser.fill('wd', 'splinter - python acceptance testing for web applications')
button = browser.find_by_xpath('//input[@type="submit"]').click()

browser.quit()
