#jh:func-base,full-type
#coding=utf-8

class myClass(object):
	def __init__(self, *args):
		'''构造函数'''
		pass

	def __str__(self):
		'''内建str方法,obj->str'''
		pass

	def __repr__(self):
		'''内建repr方法,调试时obj->str'''
		pass

	def __iter__(self):
		'''返回可迭代对象,一般返回自己,此时需要实现next方法'''
		return self

	def next(self):
		self.a += 1
		if self.a > 10000:
			raise StopIteration()
		return self.a

	def __getitem__(self, n):
		'''实现下标,索引,切片读操作'''
		if isinstance(n, int):
			#下标运算
			pass
		elif isinstance(n, slice):
			#切片运算
			start = n.start
			stop = n.stop
			pass
		else:
			pass

	def __setitem__(self, n):
		'''实现下标,索引,切片写操作'''
		pass

	def __delitem__(self, n):
		#?

	def __getattr__(self, attr):
		'''
			内建getattr方法
			实现类数据或类方法的读操作
			仅当属性没有找到时调用
		'''
		if attr == 'score':
			#类数据,返回数据
			return 100
		elif attr == 'get_score':
			#类方法,返回函数
			return lambda:100

	def __setattr__(self, attr, val):
		'''实现类数据或类方法的写操作'''
		pass

	def __delattr__(self, attr):
		#?

	def __getattribute__(self, attr):
		'''
			内建getattribute方法
			实现类数据或类方法的读操作
			无论数据或方法是否存在,都先调用该函数
			不能使用dict寻找属性,这样会导致无限递归
			要使用super(cl,self).__getattribute__(item)
			由于父类(或最终父类)没有实现__getattribute__,因此不会产生无限递归
		'''
		pass

	def getx(self):
		pass
	def setx(self, val):
		pass
	def delx(self):
		pass
	x = property(getx, setx, delx)

	@property
	def name(self):
		return self.__name
	@name.setter
	def name(self, val):
		self.__name = val
	@name.deleter
	def name(self):
		del self.__name

	def __get__(self, attr):
		'''获取对象本身'''
		pass

	def __set__(self, attr):
		'''设置对象本身'''
		pass

	def __delete__(self, attr):
		'''删除对象本身'''
		pass

	def __call__(self, *args):
		'''实现对象的调用(参数表可没有)'''
		pass

	def __new__(self, *args):
		#?

	def __del__(self, * args):
		#?

	def __unicode__(self):
		'''内建unicode方法,obj->unicode'''
		pass

	def __nonzero__(self):
		'''内建bool方法,obj->bool'''
		pass

	def __len__(self):
		'''内建len方法'''
		pass

	def __cmp__(self, obj):
		'''内建cmp'''
		pass

	def __lt__(self, obj):
		'''小于号操作符'''
		pass

	def __gt__(self, obj):
		'''大于号操作符'''
		pass

	def __eq__(self, obj):
		'''等号操作符'''
		pass

	def __le__(self, obj):
		'''小于等于操作符'''
		pass

	def __ge__(self, obj):
		'''大于等于操作符'''
		pass

	def __ne__(self, obj):
		'''非运算操作符'''
		pass

	def __add__(self, obj):
		'''加号操作符'''
		pass

	def __sub__(self, obj):
		'''减号操作符'''
		pass

	def __mul__(self, obj):
		'''乘号操作符'''
		pass

	def __div__(self, obj):
		'''除号操作符'''
		pass

	def __truediv__(self, obj):
		'''整数除法操作符'''
		pass

	def __floordiv__(self, obj):
		'''实数除法操作符'''
		pass

	def __mod__(self, obj):
		'''模运算操作符'''
		pass

	def __divmod__(self, obj):
		'''内建divmod方法'''
		pass

	def __pow__(self, obj):
		'''内建pow方法,幂操作符'''
		pass

	def __lshift__(self, obj):
		'''左移操作符'''
		pass

	def __rshift__(self, obj):
		'''右移操作符'''
		pass

	def __and__(self, obj):
		'''与操作符'''
		pass

	def __or__(self, obj):
		'''或操作符'''
		pass

	def __xor__(self, obj):
		'''异或操作符'''
		pass

	def __neg__(self):
		'''一元负操作符'''
		pass

	def __pos__(self):
		'''一元正操作符'''
		pass

	def __abs__(self):
		'''内建abs方法,绝对值'''
		pass

	def __invert__(self):
		'''取反操作符'''
		pass

	def __complex__(self, com):
		'''内建complex方法,转为复数'''
		pass

	def __int__(self):
		'''内建int方法,obj->int'''
		pass

	def __long__(self):
		'''内建long方法,obj->long'''
		pass

	def __float__(self):
		'''内建float方法,obj->float'''
		pass

	def __oct__(self):
		'''内建oct方法,转八进制'''
		pass

	def __hex__(self):
		'''内建hex方法,转十六进制'''
		pass

	def __coerce__(self, num):
		#?


#属性
#模块
__doc__#文档字符串
__name__#定义时的模块名
__file__#模块路径
__dict__#模块可用的属性名和值
#类
__doc__#文档字符串
__name__#定义时的类名
__dict__#类可用的属性名和值
__module__#包含该类的模块名
__bases__#直接父亲的元组
#对象
__dict__#对象可用的属性名和值
__class__#对象所属的类对象
#函数
__doc__#文档字符串
__name__#定义时的函数名
__module__#包含该函数的模块名
__dict__#函数可用的属性名和值
func_defaults#参数默认值元组
func_code#
func_globals#指向运行时的全局变量(没用)
func_closure#指向所引用的闭包变量的元组
#方法
__doc__#文档字符串
__name__#定义时的方法名
__module__#包含该方法的模块名
__func__#获取该方法的函数对象
__self__#返回所属对象或者所属类对象
