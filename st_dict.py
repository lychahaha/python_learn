#coding=utf-8

a = {'one':1, 'two':2}
a = dict()
a = dict(one=1, two=2)
a = dict.fromkeys(seq,val)#所有值都为val,没有提供val则是None

#获取值
print a['one']
print a.get('one')
print a.setdefault(key,val)#若键存在,则返回对应的值.否则返回val,并且插入(key,val)

#修改值
a['one'] = 2
a.update(dict1,dict2,...)#用参数字典里的键值对插入或替换,参数可以是键值对列表

#判断键是否存在
print a.has_key('one')

#遍历
for key in a:
	print key,a[key]

#删除
#a.pop(key[,default])
#没找到键,又没有默认值的时候会抛出异常
a.pop(key)#删除key对应的键值对,并返回其值
a.pop(key,default)#没找到这个键则返回默认值
a.popitem()#随机删除并返回一个键值对,字典为空时抛出异常

#dict->list
#静态
b = a.items()#键值对
b = a.keys()#键
b = a.values()#值
#动态(类似数据库对象,字典改变后它们会改变)
b = a.viewitems()#键值对
b = a.viewkeys()#键
b = a.viewvalues()#值

#dict->iter
b = a.iteritems()#键值对
b = a.iterkeys()#键
b = a.itervalues()#值
for key,val in a.iteritems():
	print key,val
for key in a.iterkeys():
	print key
for val in a.itervalues():
	print val

#拷贝
b = a.copy()#浅拷贝
b = copy.deepcopy(a)#深拷贝

#清空
a.clear()