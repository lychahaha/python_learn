#coding=utf-8

a = set()
a = set([1,2,3])

#插入
a.add(val)

#删除
a.remove(val)#没找到会抛出异常
a.discard(val)#没找到不会抛出异常
a.pop()#随机删除并返回一个元素

#集合运算
b = a.difference(iter)#差集
b = a.intersection(iter)#交集
b = a.symmetric_difference(iter)#对称差集
b = a.union(iter)#并集

a.difference_update(iter)#差集并更新
a.intersection_update(iter)#交集并更新
a.symmetric_difference_update(iter)#对称差集并更新
a.update(iter)#并集并更新

print a.isdisjoint(iter)#判断是否没有交集
print a.issubset(iter)#判断是否是子集
print a.issuperset(iter)#判断是否是父集

#拷贝
b = a.copy()#浅拷贝
b = copy.deepcopy(a)#深拷贝

#清空
a.clear()