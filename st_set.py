#jh:func-base-mixsimilar,key-type

a = set()
a = set([1,2,3])

#插入
a.add(val)
#删除
a.remove(val)#没找到会抛出异常
a.discard(val)#没找到不会抛出异常
a.pop()#随机删除并返回一个元素

#集合运算
b = a.difference(iter_obj)#差集
b = a.intersection(iter_obj)#交集
b = a.symmetric_difference(iter_obj)#对称差集
b = a.union(iter_obj)#并集
#运算符
c = a - b #差集
c = a & b #交集
c = a | b #并集
c = a ^ b #对称差集
#原地更新
a.difference_update(iter_obj)#差集并更新
a.intersection_update(iter_obj)#交集并更新
a.symmetric_difference_update(iter_obj)#对称差集并更新
a.update(iter_obj)#并集并更新

#判断
a.isdisjoint(iter_obj)#判断是否没有交集
a.issubset(iter_obj)#判断是否是子集
a.issuperset(iter_obj)#判断是否是父集

#拷贝
b = a.copy()

#清空
a.clear()