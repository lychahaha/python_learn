from appium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from appium.webdriver.common.touch_action import TouchAction

''' Appium Inspector启动语句
{
  "platformName": "Android",
  "appium:platformVersion": "10",
  "appium:deviceName": "8f877c93",
  "appium:appPackage": "com.lianjia.beike",
  "appium:appActivity": "com.lianjia.activity.MainActivity",
}
'''

# 设置初始化参数
desired_caps = dict()
desired_caps['platformName'] = 'Android' # 可以写成android
desired_caps['platformVersion'] = '10'  # 11.1.0等都可以写成11
desired_caps['deviceName'] = '1111ffff' # 设备名字可以随便写，但是不可以为空
desired_caps['appPackage'] = 'com.lianjia.beike' #app名字
desired_caps['appActivity'] ='com.lianjia.activity.MainActivity' #app入口界面
desired_caps["newCommandTimeout"] = "6000" #设置不操作自动退出的timeout（有多大设多大）
# 启动APP
driver = webdriver.Remote('http://localhost:4723/wd/hub',desired_caps)

# 查找
ele = driver.find_element(by='id', value='com.lianjia.beike:id/tv_authority_bottom')
ele = driver.find_element(by='xpath', value='/hierarchy/android.widget.FrameLayout')
ele = driver.find_elements(by='xpath', value='/hierarchy/android.widget.FrameLayout') #查找多个
## 二次查找
son_ele = ele.find_element(by='id', value='com.lianjia.beike:id/tv_authority_bottom')

# 元素属性
ele.text
ele.size #如{'height': 100, 'width': 593}
ele.location #左上角坐标,如{'x': 184, 'y': 270}

# 元素操作
## 点击
ele.click() 
## 输入框输入文字（会先清空之前的文字）
ele.send_keys('hehe')
ele.set_text('hehe')
ele.clear() #清空输入框
## 截图
ele.screenshot('a.png')


# 全局属性
driver.current_package #当前app
driver.current_activity #当前页面
driver.get_window_size() #如{'width':1080,'height':1920}
driver.network_connection #4表示在使用流量，2表示在使用wifi，1表示飞行模式
driver.battery_info #{'level': 1, 'state': 2}
driver.is_keyboard_shown() #判断键盘是否显示
driver.is_app_installed('com.lianjia.beike') #判断app是否已安装

# 全局操作
## 点击（同时点击，最多五个点）
driver.tap([(100,200),(400,300)], duration=500)
## 拖动（都会因为延迟而有误差）
driver.swipe(start_x=100, end_x=100, start_y=1000,end_y=100, duration=1000)
driver.scroll(ele1, ele2, duration=1000)
driver.drag_and_drop(ele1, ele2)
## 摇一摇
driver.shake()
## 截图
driver.get_screenshot_as_file('a.png')
## 返回
driver.back()
## 文件传输
s = driver.pull_file('/xxx/xxx') #手机->电脑
driver.push_file('/xxx/xxx', s) #电脑->手机
## 打开通知栏
driver.open_notifications()
## app相关
driver.close_app() #关闭app
driver.install_app('C:\\a.apk') #安装app
driver.remove_app('com.lianjia.beike') #卸载app
## 退出
driver.quit()
## 熄屏和亮屏
driver.lock()
driver.unlock()
## 隐藏键盘
driver.hide_keyboard()
## 键盘码
driver.press_keycode(3)
'''
3:home键
24:增加音量键
26:power键
187,82:调出APP菜单
220,221:屏幕变暗/变亮
'''


# 复杂动作
action = TouchAction(driver)
action.press(x=105,y=400).perform()
## 种类
action.tap(ele) #轻触
action.tap(x=100,y=200) #(下面大部分操作均可支持两种方式)
action.press(x=100,y=200) #按下
action.long_press(x=100,y=200,duration=1000) #长按
action.release() #放开
action.move_to(x=100,y=200) #移动
action.wait(1000) #等待


# wait同步
wait = WebDriverWait(driver, 30)
ele = wait.until(EC.presence_of_element_located(('id', "com.tencent.mm:id/cp_")))
