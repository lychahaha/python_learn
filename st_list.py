#jh:func-base-mixsimilar,key-type
#coding=utf-8

a = [1,3,2,4,5]
a = list()
a = list(iter)

#插入
a.insert(ix,val)
a.append(val)#尾插
a.extend(seq)#尾插一个列表

#查找
a.index(val)#没有找到会抛出异常
a.index(val,beg,end)#给定范围

#删除
a.remove(val)#找不到会抛出异常
a.pop()#删除并返回最后一个元素
a.pop(ix)#删除并返回给定下标的元素

#计数
print a.count(val)

#排序
#a.sort(cmp=None,key=None,reverse=False)
a.sort()
a.sort(fx_cmp)#自定义cmp函数,cmp(x,y)->(-1,0,1)
a.sort(fx_cmp,fx_key)#自定义要比较的对象,key(obj)->obj
a.sort(fx_cmp,fx_key,True)#倒序

#倒转
a.reverse()