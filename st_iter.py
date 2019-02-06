#jh:func-base-mixsimilar,full-type
from itertools import *
import functools

#range
ite = range(3) #0,1,2
ite = range(1,3) #1,2
ite = range(1,8,2) #1,3,5,7
#islice
#range从自然数到可迭代对象的一般化
ite = islice('abcde', 2) #a,b
ite = islice('abcde', 2, 4) #c,d
ite = islice('abcde', 2, None) #c,d,e
ite = islice('abcde', 2, None, 2) #c,e


#count
count() #自然数0,1,2,3,...
count(3) #3,4,5,...
count(3,2) #3,5,7,9,...
#cycle
ite = cycle('abc') #a,b,c,a,b,c,a,...
#repeat
ite = repeat('a') #a,a,a,a,...
ite = repeat('a', 4) #a,a,a,a
#takewhile
#变成false就停
ite = takewhile(lambda x:x<=10, count(6)) #6,7,8,9,10
#dropwhile
#变成false才开始
ite = dropwhile(lambda x:x<=10, range(4,13)) #11,12


#map
ite = map(lambda x:x*x, [1,2,3]) #1,4,9
#starmap
#map是一对一,starmap扩展到多对一,*args->ret
ite = starmap(pow, [(2,3),(1,2),(3,0)]) #8,1,1


#zip
ite = zip([1,2],[3,4],[5,6]) #(1,3,5),(2,4,6)
ite = zip([1,3,5],[2,4,6]) #(1,2),(3,4),(5,6)
#zip_longest
#长度不等时填充默认值
ite = zip_longest('ab','123') #(a,1),(b,2),(None,3)
ite = zip_longest('ab','123', fillvalue='?') #(a,1),(b,2),(?,3)


#reduce
val = functools.reduce(lambda x,y:x+y, [1,2,3]) #1+2+3=6
val = functools.reduce(lambda x,y:x+y, [1,2,3], 2) #2+1+2+3=8
#accumulate
ite = accumulate([1,2,3]) #1,3,6


#groupby
#相邻重复元素放在一起
#每项右边其实是ite,要list(ite)才是下面注释的样子
#func的返回值仅用来判断是否相等,迭代的时候还是原本的东西
ite = groupby('AABCCC') #('A',['A','A']),('B',['B']),('C',['C','C','C'])
ite = groupby('AaBcCC', lambda c:c.upper())


#compress
#掩码筛选迭代
ite = compress('abc',[1,0,1]) #a,c
#filter
#过滤器筛选迭代
ite = filter(lambda x:x%2, [1,2,3]) #1,3
ite = filter(None, [1,2,3]) #None则使用bool函数,1,2,3
#filterfalse
ite = filterfalse(lambda x:x%2, [1,2,3]) #2


#chain
#多个迭代器放在一起迭代
ite = chain('123','abc') #1,2,3,a,b,c
ite = chain(d1,d2,d3,d4)
#product
#笛卡尔积,多层for循环嵌套
ite = product('ab','cd') #(a,c),(a,d),(b,c),(b,d)
ite = product(d1, d2, d3, d4)
ite = product('01', 3) # product('01','01','01')
ite = product('ab','cd', 2) #product('ab','cd','ab','cd')
#permutations
#全排序(组合排列)
ite = permutations('abc') #(a,b,c),(a,c,b),(b,a,c),(b,c,a),(c,a,b),(c,b,a)
ite = permutations('abc', 2) #(a,b),(a,c),(b,a),(b,c),(c,a),(c,b)
#combinations
#组合
ite = combinations('abc', 2) #(a,b),(a,c),(b,c)
#combinations_with_replacement
#包括重复自己的组合
ite = combinations_with_replacement('abc', 2) #(a,a),(a,b),(a,c),(b,b),(b,c),(c,c)


#tee
#返回n个ite的副本(用来缓存,即实际上只有一次的生成)
#原本的ite在副本使用期间不能使用
ite1,ite2 = tee(ite, n=2)