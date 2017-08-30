#coding=utf-8

from mpi4py import MPI

#通信组核心对象
comm = MPI.COMM_WORLD

#获取进程rank(id)
print comm.Get_rank()

#获取通信组进程数
print comm.Get_size()

#点对点
'''
在发送时,假若数据很少,发送采取异步式;假若数据很多,发送采取同步式
接收均采用同步式
'''
#发送
comm.send(data, dest=1)
#接收
data = comm.recv(source=1)
#发送+接收
data = comm.sendrecv(data, dest=1)
#缓冲模式发送
comm.bsend(data, dest=1)
#同步模式发送
comm.ssend(data, dest=1)


#广播
'''
发送是异步式,接收是同步式
'''
#发送(既发送又接收)
comm.bcast(data, root=0)#返回值也是这个data,可不写
#接收
data = comm.bcast(None, root=0)

#散播
#发送(既发送又接收)
data = comm.scatter(data, root=0)
#接收
data = comm.scatter(None, root=0)

#收集
#发送
comm.gather(data, root=0)#返回none
#接收(既发送又接收)
data = comm.gather(data, root=0)

#规约(聚合)
#发送
comm.reduce(data, root=0, op=MPI.SUM)#返回none
#接收(既发送又接收)
data = comm.reduce(data, root=0, op=MPI.SUM)

#收集+广播
data = comm.allgather(data)

#规约+广播
data = comm.allreduce(data, op=MPI.SUM)

#转置
data = comm.alltoall(data)

#前置规约(前i个规约的结果发给进程i)
data = comm.scan(data, op=MPI.SUM)


#组
#创建
new_comm = comm.Create(group)
#分割
new_comm = comm.Split(color)

#获取通信组rank
group.Get_rank()
#获取通信组size
group.Get_size()

#挑选
group.Incl([1,2,3])
#剔除
group.Excl([1,2,3])
#交集
group = MPI.Group.Intersection(group1, group2)
group = MPI.Group.Intersect(group1, group2)
#并集
group = MPI.Group.Union(group1, group2)
#差集
group = MPI.Group.Difference(group1, group2)
