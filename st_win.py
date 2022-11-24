import win32con
import win32api
import win32gui

# 移动鼠标
win32api.SetCursorPos([30,150])
# 鼠标左右键单击
win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP|win32con.MOUSEEVENTF_LEFTDOWN, 0,0,0,0)
win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP|win32con.MOUSEEVENTF_RIGHTDOWN, 0,0,0,0)
# 鼠标滚动
win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL,0,0,-1) #-1表示往下滚，1表示往上滚
# 键盘点击
win32api.keybd_event(key, 0, 0, 0) #按下
win32api.keybd_event(key, 0, win32con.KEYEVENTF_KEYUP, 0) #释放

# 获取窗口句柄
hid = win32gui.FindWindow(None, 'title')
# 获取子空间句柄
sid = win32gui.FindWindowEx(hid, None, 'button', '确定')
# 最大化窗口
win32gui.ShowWindow(hid, win32con.SW_SHOWMAXIMIZED)
# 最小化窗口
win32gui.ShowWindow(hid, win32con.SW_SHOWMINIMIZED)
# 获取窗口位置
left,top,right,bottom = win32gui.GetWindowRect(hid)
# 获取窗口标题
title = win32gui.GetWindowText(hid)
# 获取窗口类名
clsname = win32gui.GetClassName(hid)
# 点击按钮
win32gui.SendMessage(hid, win32con.BM_CLICK, 0, 0)
# 关闭窗口
win32gui.PostMessage(hid, win32con.WM_CLOSE, 0, 0)


