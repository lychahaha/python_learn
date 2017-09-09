#coding=utf-8
import logging

#输出
logging.debug('debug msg')
logging.info('info msg')
logging.warning('warning msg')
logging.error('error msg')
logging.critical('critical msg')

#输出设置
logging.basicConfig(level=logging.DEBUG,
	format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
	datefmt='%a, %d %b %Y %H:%M:%S',
	filename='my.log',
	filemode='w')
'''
level:日志级别

format:输出格式
%(name)s: Logger的名字
%(levelno)s: 打印日志级别的数值
%(levelname)s: 打印日志级别名称
%(pathname)s: 打印当前执行程序的路径，其实就是sys.argv[0]
%(filename)s: 打印当前执行程序名
%(module)s: 调用日志输出函数的模块名
%(funcName)s: 打印日志的当前函数
%(lineno)d: 打印日志的当前行号
%(created)f: 当前时间，用UNIX标准的表示时间的浮点数表示
%(relativeCreated)d: 输出日志信息时的，自Logger创建以来的毫秒数
%(asctime)s: 打印日志的时间
%(thread)d: 打印线程ID
%(threadName)s: 打印线程名称
%(process)d: 打印进程ID
%(message)s: 打印日志信息

datefmt:时间格式,同time.strftime()

filename:日志文件名

filemode:打开模式,'w'或'a'

stream:输出流,sys.stderr或sys.stdout,有filename时被忽略
'''

#流句柄
#让信息同时输出到文件和屏幕
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

##多log多输出
#创建日志
l1 = logging.getLogger("l1")
l2 = logging.getLogger("l2")



#创建文件句柄,流句柄(如屏幕)
fh = logging.FileHandler("test.log")
ch = logging.StreamHandler()

#设置句柄的格式
formatter = logging.Formatter("%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

#设置句柄的等级
fh.setLevel(logging.WARNING)



#创建过滤器
filt = logging.Filter("a.b.c")#(只有a.b.c前缀的日志才可以输出)

#增/删过滤器
fh.addFilter(filt)
fh.removeFilter(filt)



#设置日志的输出句柄
l1.addHandler(fh)
l1.addHandler(ch)

#设置日志等级
#(日志和句柄各有等级,先检查日志后检查句柄)
l1.setLevel(logging.WARNING)

#增/删过滤器
#(日志和句柄各有过滤器,先过滤日志后检查句柄)
l1.addFilter(filt)
l1.removeFilter(filt)

#log
l1.warning("haha!")
l2.error("error!!")

'''
Logger:日志类
FileHandler,StreamHandler:文件句柄,流句柄
Formatter:格式类
'''