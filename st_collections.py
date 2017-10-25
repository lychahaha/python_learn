#jh:class-base,lazy-type
from collections import *

#计数器(可以当成字典用)
c = Counter(l)
##top-k
#返回(ele,cnt)的列表
c.most_common(k)
##再添加一个列表参与计数
c.update(l)


#双端队列(可以当成列表用)
d = deque()
d = deque(l)
##push and pop
d.append(x)
d.appendleft(x)
x = d.pop()
x = d.popleft()
d.extend(l)
d.extendleft(l)



