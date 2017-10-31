#jh:class-base,key-type
#coding=utf-8
import threading

#子线程函数
def fx(*args):
	print args

#创建子线程
t = threading.Thread(target=fx, args=('a','b'), name='xxx')

#子线程类
class myThread(threading.Thread):
	def __init__(self, *args):
		super(myThread, self).__init__()
		self.args = args

	#子线程函数
	def run(self):
		print self.args

t = myThread('a', 'b')

t.start() #启动子线程
t.join() #等待子线程
t.join(1) #设置timeout(无法判断能否等待成功)

##设置成守护线程,父死子也死(需在start前设置)
t.daemon
t.isDaemon()
t.setDaemon(True)
##名字
t.name
t.getName()
t.setName()
##id
##只在start后有效,否则返回none
t.ident
##是否活跃
t.is_alive()
t.isAlive()



#当前活跃线程数(包括主线程)
threading.active_count()
threading.activeCount()

#当前线程对象
threading.current_thread()
threading.currentThread()

#当前活跃线程对象列表(包括主线程)
threading.enumerate()

#装饰函数
#所有线程启动前都要调用该函数
threading.settrace(func)
#所有线程结束前都要调用该函数
threading.setprofile(func)



#锁
lock = threading.Lock()
##获取锁
lock.acquire()
##释放锁
lock.release()


#RLock
#同一线程能多次acquire(),但需同样次数的release()
rlock = threading.RLock()
rlock.acquire()
rlock.release()


#Condition(高级锁)
cdt = threading.Condition()
cdt.acquire()
cdt.release()
##wait
##释放锁,同时被挂起.当接收到通知或超时才唤醒
cdt.wait()
cdt.wait(5)
##notify
cdt.notify() ##唤醒一个挂起的线程
cdt.notify_all() ##唤醒所有挂起的线程
cdt.notifyAll()


#事件event
#自旋锁
e = threading.Event()
##wait
##等待,直到标识位为true或超时才唤醒
e.wait()
e.wait(5)
##标识设置
e.set()
e.clear()
e.isSet()


#计时器
t = threading.Timer(3, func)
t.start()#3秒后运行func


#线程不共享全局变量(thread->var)
d = threading.local()
d.k = 1


#信号量
s = threading.Semaphore(5)
s.acquire()
s.release()

