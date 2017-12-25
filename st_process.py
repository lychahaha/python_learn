#jh:class-base,key-type
#coding=utf-8
import multiprocessing

#子进程函数
def fx(*args):
	print args

#创建子进程
p = multiprocessing.Process(target=fx, args=(3,4))


#子进程类
class myProcess(multiprocessing.Process):
	def __init__(self, *args):
		super(myProcess, self).__init__()
		self.args = args

	#子进程函数
	def run(self):
		print self.args

p = myProcess('a', 'b')

##属性
p.name #名字
p.pid #id
p.daemon #是否设置成守护进程(父死子也死,要在start前设置)
p.exitcode #返回值,运行时为None,为-N表示被信号N结束
##方法
p.start() #启动子进程
p.join() #等待子进程
p.terminate() #杀死子进程
p.is_alive() #是否活跃



#CPU数量
multiprocessing.cpu_count()

#当前活跃进程对象列表(不包括主进程)
multiprocessing.active_children()



#下面的类都可以通过参数传递,进程间共享

#伪共享内存的变量
manager = multiprocessing.Manager()
d = manager.dict({'1':1,'2':2})
l = manager.list([2,3,3])


#锁
##可以使用with lock上下文,代替acquire和release
lock = multiprocessing.Lock()
lock.acquire() #获取锁
lock.release() #释放锁


#队列
q = multiprocessing.Queue()
q = multiprocessing.Queue(10) #队列最大长度
##进队
##如果队满,且block为false,则抛出异常
##如果为true,则在超时后才抛出异常
q.put(data, block=True, timeout=1)
##出队
##参数和进队相似
data = q.get(blcok=True, timeout=1)
##size(结果不可靠)
q.empty()
q.full()
q.qsize()


#进程池
pool = multiprocessing.Pool(processes=10)
##分配任务
pool.apply_async(func, (a,b,c)) #非阻塞
pool.apply(func, (a,b,c)) #进程运行完才返回
ret = pool.map(func, ((a1,b1),(a2,b2))) #类似map
##结束
pool.close() #不再运行新进程
pool.join() #要在close后调用
pool.terminate() #kill


#事件event
#自旋锁
e = multiprocessing.Event()
##wait
##等待,直到标识位为true或超时才唤醒
e.wait()
e.wait(5)
##标识设置
e.set()
e.clear()
e.isSet()


#信号量
s = multiprocessing.Semaphore(5)
s.acquire()
s.release()


#管道
##返回(p1,p2)
##如果duplex为true,则是全双工,即p1,p2均可收发
##如果为false,则p1为接收,p2为发送
p1,p2 = multiprocessing.Pipe(duplex=True)
##发送和接收
##如果管道已关闭,recv会抛出异常
p2.send(data)
data = p1.recv()

