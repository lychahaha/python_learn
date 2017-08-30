#coding=utf-8
import time
import datetime

#获取时间戳(浮点数)
now = time.time()
print now

#时间戳转结构体时间
print time.localtime(now)#当地时间
print time.gmtime(now)#转标准时间(以now为localtime)

now = time.localtime(now)

#时间转字符串
print time.asctime(now)
print time.strftime("%y %m %d %H:%M:%S", now)#两位数年份
print time.strftime("%Y", now)#四位数年份
print time.strftime("%I %p", now)#12小时制的小时,PM or AM
print time.strftime("%a %b %d", now)#星期(英文),月份(英文),日期(数字)
print time.strftime("%c", now)#标准日期时间输出

#字符串转时间
print time.strptime("2016 07 01 09:30:00", "%Y %m %d %H:%M:%S")

#结构体时间转时间戳
print time.mktime(now)


#程序执行时间(运行态)
t0 = time.clock()