# 启动关闭服务
adb start-server
adb kill-server

# 检查设备是否得到调试权限
adb devices

# 模拟点击
adb shell input tap {x} {y}

# 模拟拖拽
## 参数是起终点坐标和拖拽时间
adb shell input swipe {x1} {y1} {x2} {y2} {time}

# 屏幕截图
adb shell screencap /sdcard/a.png
adb shell screencap -p # 输出到标准输出(二进制码)

# 执行linux命令
adb shell {cmd}