#jh:class-base,key-type
import time
import datetime

#时间戳(浮点数)
timestamp = time.time()
##时间元组->时间戳
timestamp = time.mktime(tt)

#时间元组(time.struct_time,继承元组)
tt = time.localtime()
tt = time.gmtime() #标准时间
##属性
tt.tm_year
tt.tm_mon
tt.tm_mday
tt.tm_yday #一年的第几天
tt.tm_wday #星期几
tt.tm_hour
tt.tm_min
tt.tm_sec
##时间戳->时间元组
tt = time.localtime(timestamp)
tt = time.gmtime(timestamp) #以timestamp为localtime
##时间字符串->时间元组
tt = time.strptime(s, '%Y-%m-%d %H:%M:%S')

#时间字符串
s = time.asctime()
##时间戳->时间字符串
s = time.ctime(timestamp) #'Sun Jan 21 16:59:32 2018'
##时间元组->时间字符串
s = time.asctime(tt) #格式同time.ctime
##时间元组->时间字符串(带格式)
s = time.strftime('%Y-%m-%d %H:%M:%S', tt)
'''
%Y:year
%m:month
%d:day
%H:hour
%M:minute
%S:second
%z:相对标准时间的偏移量
%a/%A:星期英文缩写/不缩写
%b/%B:月份缩写/不缩写
%c:time.asctime的形式
%I:12小时制的小时
%p:AM/PM
'''
##datetime->时间字符串
s = datetime.datetime.strftime('%Y-%m-%d %H:%M:%S', dt)

#datetime.datetime
dt = datetime.datetime.now()
##属性
dt.year
dt.month
dt.day
dt.hour
dt.minute
dt.second
dt.microsecond
##方法
dt.isocalendar() #返回(year,这年的第几周,星期几)
dt.isoweekday() #星期几
##时间字符串->datetime
dt = datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
##日期加减
dt2 = dt + datetime.timedelta(days=1, seconds=-10, microseconds=1000)

#睡眠
time.sleep(0.5)#0.5s

#程序执行时间(float)
t = time.clock()

#相对标准时间的偏移秒数(int)
time.altzone

