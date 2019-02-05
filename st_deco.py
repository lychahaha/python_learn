#jh:proc-base,lazy-type

#LEGB
#L:local
#E:enclosing
#G:global
#B:build-in




#函数里包裹一层函数

def my_sum(*arg):
	return sum(arg)

def my_avg(*arg):
	return sum(arg)/len(arg)

def deco(func):
	def in_deco(*arg):
		if len(arg) == 0:
			return 0
		for val in arg:
			if not isinstance(val, int):
				return 0
		return func(*arg)
	return in_deco

my_sum = deco(my_sum)
my_avg = deco(my_avg)

print(my_sum(1,2,3,4,5))
print(my_avg(1,2,3,4,5))




#使用装饰器这个语法糖
'''
@deco
def fx(...):
	...
1.装饰器deco实际上是一个func->func的函数
2.输入的func是fx,deco的输出会覆盖旧的fx作为新的fx
3.deco应该要保证新旧fx的参数列表保持不变
'''

def deco(func):
	def in_deco(*arg):
		if len(arg) == 0:
			return 0
		for val in arg:
			if not isinstance(val, int):
				return 0
		return func(*arg)
	return in_deco

@deco
def my_sum(*arg):
	return sum(arg)

@deco
def my_avg(*arg):
	return sum(arg)/len(arg)

print(my_sum(1,2,3,4,5))
print(my_avg(1,2,3,4,5))




#类里的装饰器

def deco(func):
	def in_deco(self, x):
		if y == 0:
			return 0
		return func(self, x)
	return in_deco

class XXX(object):
	@deco
	def fx(self, x)
		return 1/x

xxx = XXX()
print(xxx.fx(6))
print(xxx.fx(0))




#装饰器带参数
'''
@deco_maker(args)
def fx(...):
	...
1.装饰器工厂deco_maker实际上是一个args->deco func的函数
2.
	1)deco_maker接受args输入,返回一个deco装饰器函数
	2)deco装饰器函数再接受fx输入,返回一个新的fx
'''

def deco_maker(name):
	def deco(func):
		def in_deco(*args, **kw):
			print(name, len(args), len(kw))
			return func(*args, **kw)
		return in_deco
	return deco

@deco_maker('jh')
def fx1(k):
	return k*k

@deco_maker('ji')
def fx2(k):
	return k+1

