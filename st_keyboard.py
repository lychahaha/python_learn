'''
安装pyhook的whl
替代pyhook：pip install pywinhook

pip install pyuserinput
'''

from pykeyboard import PyKeyboard
from pymouse import PyMouse
import pyHook
#import pyWinhook as pyHook
import pythoncom



#typical

# 模拟鼠标和键盘操作
keyboard = PyKeyboard()
mouse = PyMouse()

keyboard.press_key('a')
mouse.click(200, 200, 1)


# hook鼠标和键盘
def key_down_func(event):
    if event.Key == 'A':#必须大写
        pass
    return True

pw = pyHook.HookManager() #创建hook管理器
pw.KeyDown = key_down_func #设置事件处理函数
pw.HookKeyboard() #设置hook
pythoncom.PumpMessages() #进入消息循环



# full

# 模拟鼠标操作
mouse.click(x, y, button=1) #点击。第三个参数：1是左键，2是右键，3是中键
mouse.press(x, y, 1) #按下
mouse.release(x, y, 1) #释放
mouse.move(x, y) #移动
mouse.drag(x, y) #按下，移动，释放
mouse.scroll(vertical=3, horizontal=None) #滚动，可以同时垂直和水平滚动。v为正数是往上滚动。
## 获取信息
x,y = mouse.position() #获取鼠标位置
w,h = mouse.screen_size() #获取屏幕尺寸

# 模拟键盘操作
keyboard.press_key('a') #按下
keyboard.release_key('a') #释放
keyboard.press_keys([keyboard.control_key,'a']) #组合键
keyboard.type_string('abcef') #键入字符串
##特殊键
keyboard.backspace_key
keyboard.tab_key
keyboard.enter_key
keyboard.shift_key
keyboard.control_key
keyboard.alt_key
keyboard.escape_key
keyboard.left_key
keyboard.right_key
keyboard.up_key
keyboard.down_key
keyboard.function_keys[1] #F1




# event对象和事件处理函数

## 鼠标事件处理函数
def OnMouseEvent(event):
  print('MessageName:',event.MessageName)  #事件名称
  print('Message:',event.Message)          #windows消息常量 
  print('Time:',event.Time)                #事件发生的时间戳        
  print('Window:',event.Window)            #窗口句柄         
  print('WindowName:',event.WindowName)    #窗口标题
  print('Position:',event.Position)        #事件发生时相对于整个屏幕的坐标
  print('Wheel:',event.Wheel)              #鼠标滚轮
  print('Injected:',event.Injected)        #判断这个事件是否由程序方式生成，而不是正常的人为触发。

  # 返回True代表将事件继续传给其他句柄，为False则停止传递，即被拦截
  return True

## 键盘事件处理函数
def OnKeyboardEvent(event):
  print('MessageName:',event.MessageName)          #同上
  print('Message:',event.Message)
  print('Time:',event.Time)
  print('Window:',event.Window)
  print('WindowName:',event.WindowName)
  print('Ascii:', event.Ascii, chr(event.Ascii))   #按键的ASCII码
  print('Key:', event.Key)                         #按键的名称
  print('KeyID:', event.KeyID)                     #按键的虚拟键值
  print('ScanCode:', event.ScanCode)               #按键扫描码
  print('Extended:', event.Extended)               #判断是否为增强键盘的扩展键
  print('Injected:', event.Injected)
  print('Alt', event.Alt)                          #是某同时按下Alt
  print('Transition', event.Transition)            #判断转换状态?

  return True

# 事件
## 常用
pw.MouseAllButtonsDown
pw.KeyDown
## 全部
### 鼠标
MouseMove
MouseWheel
MouseLeftUp
MouseLeftDown
MouseLeftDbl #双击
MouseRightUp
MouseRightDown
MouseRightDbl
MouseMiddleUp
MouseMiddleDown
MouseMiddleDbl
MouseAllButtonsUp # =left_up+right_up+mid_up
MouseAllButtonsDown
MouseAllButtonsDbl
MouseAllButtons # =all_up+all_down+all_dbl
MouseAll #=all_buttons+move+wheel
### 键盘
KeyUp
KeyDown
KeyChar #?
KeyAll #up+down+char

# 设置hook
pw.HookMouse()
pw.HookKeyboard()

# 取消hook
pw.UnhookMouse()
pw.UnhookKeyboard()
