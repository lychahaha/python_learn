#coding=utf-8
import multiprocessing

#子进程函数
def fx(*args):
	print args

#创建子进程
p = multiprocessing.process(target=fx, args=(3,4))

#子进程类
class myProcess(multiprocessing.Process):
	def __init__(self, *args):
		super(myProcess, self).__init__()
		self.args = args

	#子进程函数
	def run(self):
		print self.args

p = myProcess('a', 'b')

#是否设置成守护进程(父死子也死)
p.daemon

#启动子进程
p.start()

#等待子进程
p.join()

#名字
p.name

#id
p.pid

#是否活跃
p.is_alive()

#CPU数量
multiprocessing.cpu_count()

#当前活跃进程对象列表(不包括主进程)
multiprocessing.active_children()

#锁
lock = multiprocessing.Lock()
#获取锁
lock.acquire()
#释放锁
lock.release()

#事件event
#自旋锁
e = multiprocessing.Event()

#wait
#等待,直到标识位为true或超时才唤醒
e.wait()
e.wait(5)

#标识设置
e.set()
e.clear()
e.isSet()

#信号量
s = multiprocessing.Semaphore(5)
s.acquire()
s.release()

#队列
q = multiprocessing.Queue()

#进队
#如果队满,且block为false,则抛出异常
#如果为true,则在超时后才抛出异常
q.put(data, block=True, timeout=1)

#出队
#参数和进队相似
data = q.get(blcok=True, timeout=1)

#管道
#返回(p1,p2)
#如果duplex为true,则是全双工,即p1,p2均可收发
#如果为false,则p1为接收,p2为发送
p = multiprocessing.Pipe(duplex=True)

#发送和接收
#如果管道已关闭,recv会抛出异常
p2.send(data)
data = p1.recv()

