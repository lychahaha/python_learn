#jh:class-base,lazy-type
from collections import *

'''计数器(可以当成字典用)'''
c = Counter([1,2,3,2,4,1])
c = Counter('43242523423')
c = Counter(a=3, b=4)
c = Counter({'a':3,'b':2})

##top-k
#返回(ele,cnt)的列表
c.most_common(k)
c.most_common() #返回全部

##再添加一个列表/字符串...参与计数
c.update(l)
##反向计数(可以变到负)
c.subtract(l)


'''双端队列(可以当成列表用,主要是时间复杂度不同)'''
d = deque()
d = deque(l)

##push
d.append(x)
d.appendleft(x)
d.extend(l)
d.extendleft(l)
##pop
x = d.pop()
x = d.popleft()

##rotate
d.rotate(3) #把队尾的3个元素旋转到队首
d.rotate(-3) #把队首的3个元素旋转的队尾


'''命名元组(可以认为是快速写结构体)'''
Point = namedtuple('Point', ['x','y'])
p = Point(3,4)

##索引
p[0]
p.x

##下面和tuple一样
p.count(3)
p.index(3)


'''多个字典合并成一个字典(可以认为是list(dict))'''
#d不会深拷贝d1和d2,所以d1,d2修改后d也会修改
d1 = {1:1,2:2}
d2 = {1:4,3:3}
d = ChainMap(d1,d2)

##读(从前往后查询)
d[1] #1
d[3] #3
##写(只修改第一个dict,包括其他pop之类的修改操作)
d[1] = 2

##获取所有字典
d.maps #[d1,d2]

##增加新字典(放在最前)
d = d.new_child()
d = d.new_child(d0)
##删除字典(只能删除最前)
d = d.parents


'''继承'''
#最好继承它们,而不是直接继承list,dict,str
UserList,UserDict,UserString