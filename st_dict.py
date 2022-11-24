#jh:func-base-mixsimilar,key-type

a = {'one':1, 'two':2}
a = dict()
a = dict(one=1, two=2)
a = dict.fromkeys(seq,val)#所有值都为val,没有提供val则是None

#获取值
x = a['one']
x = a.get('one') #没有这个键则返回None
x = a.get('one', 0) #没有这个键则返回0
x = a.setdefault(key,val)#若键存在,则返回对应的值.否则返回val,并且插入(key,val)

#修改值
a['one'] = 2
a.update(dict1,dict2,...)#用参数字典里的键值对插入或替换,参数可以是键值对列表

#判断键是否存在
'one' in a

#遍历
for key in a:
	print(key,a[key])

#删除
#a.pop(key[,default])
#没找到键,又没有默认值的时候会抛出异常
a.pop(key)#删除key对应的键值对,并返回其值
a.pop(key,default)#没找到这个键则返回默认值
a.popitem()#随机删除并返回一个键值对,字典为空时抛出异常
a.clear()#清空

#dict->list
b = list(a.items())#键值对
b = list(a.keys())#键
b = list(a.values())#值


from collections import defaultdict,OrderedDict

#defaultdict
#自动插入新键的字典
d = defaultdict(list) #值是一个列表
d = defaultdict(int) #值默认是0
d[3].append(4) #没有3时自动插入(3,[])
d[4] += 1 #没有4时自动插入(4,0)


#OrderedDict
#键的遍历按插入顺序排序的字典(相当于可快速索引的队列)
d = OrderedDict()

d.move_to_end(key) #把key从队列中拿出,并放到队尾
d.move_to_end(key, last=False) #放到队首
