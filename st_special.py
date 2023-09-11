#jh:func-base,full-type
#coding=utf-8

class MyClass(object):
	def __init__(self, *args, **kw):
		'''MyClass(), 构造函数'''
		pass

	def __new__(self, *args, **kw)
		pass#?

	def __del__(self):
		pass#?




	def __str__(self):
		'''str(k)'''
		pass

	def __repr__(self):
		'''repr(k),调试时调用'''
		pass

	def __len__(self):
		'''len(k)'''
		pass




	'''
	for i in xx:
		...
	xx.__iter__返回一个有__next__方法的对象xxx,一般返回自己
	每次迭代都会调用xxx.__next__方法,得到下一个值,迭代完则抛出StopIteration异常

	__iter__和__next__对应内建函数iter(xx)和next(xx)
	'''
	def __iter__(self):
		'''iter(k)'''
		self.a = 0
		return self

	def __next__(self):
		'''next(k)'''
		self.a += 1
		if self.a > 10000:
			raise StopIteration()
		return self.a
	
	'''
	iter有两个参数时,直接调用iter_obj.__call__(不带任何参数)
	当iter_obj.__call__返回等于stop的值,外部会抛出StopIteration异常,循环结束

	next有两个参数时,cache到StopIteration异常,会返回default(后面继续调用也继续返回)
	'''
	iter(iter_obj, stop)
	next(iter_obj, default)




	def __call__(self, *args, **kw):
		'''k(*args, **kw) ,实现对象的调用)'''
		pass




	def __getitem__(self, key):
		'''k[key], 实现索引切片读操作'''
		if isinstance(key, int):
			#下标运算
			pass
		elif isinstance(key, slice):
			#切片运算
			start,stop,step = key.start,key.stop,key.step
			pass

	def __setitem__(self, key, val):
		'''k[key]=val, 实现索引切片写操作'''
		pass

	def __delitem__(self, key):
		pass#?

	def __contains__(self, key):
		'''key in k'''
		pass

	def __missing__(self, key):
		'''k[key], 当这个键不存在时调用'''
		pass#?




	def __enter__(self):
		'''上下文进入时调用,返回一个对象'''
		pass

	def __exit__(self, exc_type, exc_value, traceback):
		'''上下文离开时调用'''
		pass




	'''
	关于__getattr__, __setattr__, __getattribute__, property
	get attribute流程
		1	调用类的__getattribute__
		2.1	如果用户觉得非法,可以在这里卡住(返回None等等)
		2.2	调用super().__getattribute__(一般会传到object)
		2.3 报AttributeError错误,直接跳到__getattr__(参考4)
		3	object.__getattribute__里判断
		3.1	如果本身有这个属性,则返回这个属性,多级返回到类的__getattribute__
		3.2	如果有这个property,则调用它的getter,再多级返回到类的__getattribute__
		3.3 如果都没有,则报AttributeError错误
		4.1	类的__getattribute__往上返回,完成操作
		4.2	上面所有过程类会cache AttributeError,然后调用__getattr__
		5	__getattr__直接返回一个值(不会再经过__getattribute__),完成操作
	set attribute流程
		1	调用__setattr__
		2.1	如果用户觉得非法,可以在这里卡住(返回None等等)
		2.2	调用super().__setattr__(一般会传到object)
		3	object.__setattr__里判断
		3.1	如果本身有这个属性,则直接赋值
		3.2	如果有这个property,则调用它的setter
		4	经多级返回到类的__setattr__,再往上返回,完成操作
	'''

	def __getattr__(self, attr):
		pass

	def __setattr__(self, attr, val):
		pass

	def __delattr__(self, attr):
		pass#?

	def __getattribute__(self, attr):
		pass


	'''
	property有下面两种方式实现
	'''

	def getx(self):
		pass
	def setx(self, val):
		pass
	def delx(self):
		pass
	x = property(getx, setx, delx)

	@property
	def x(self):
		pass
	@x.setter
	def x(self, val):
		pass
	@x.deleter
	def x(self):
		pass


	'''
	getattr和setattr相当于obj.name和obj.name=val
	就是说getattr也会进入obj.__getattribute__,参与attribute流程
	obj.__getattr__抛出AttributeError异常,getattr会cache并返回default(除非没default)
	hasattr通过是否抛出AttributeError异常来判断有没有这个属性
	'''
	getattr(obj, name, default)
	setattr(obj, name, val)
	hasattr(obj, name)


	'''
	这个类的对象被某个类owner拥有时
	当owner的某个实例instance访问到这个对象时
	当attribute流程执行到object.__getattribute__和objest.__setattr__时
	会分别调用__get__和__set__

	它的定位是用来代替property
	把property的get,set,del函数封装到一个单独的类里
	与owner分离开
	'''

	def __get__(self, instance, owner):
		pass

	def __set__(self, instance, value):
		pass

	def __delete__(self, instance):
		pass#?




	def __dir__(self):
		'''dir(k)'''
		pass




	def __reversed__(self):
		'''reversed(k)'''
		pass




	def __hash__(self):
		'''
		hash(k)
		dict,set等需要唯一标记一个对象,它们会调用hash(k)
		如果发现两个hash一样,则会调用__eq__

		k.__hash__返回一个int作为hash值
		定义__hash__,需要先定义__eq__
		'''
		pass




	@classmethod
	def __instancecheck__(cls, instance):
		'''isinstance(instance, class)'''
		pass#?

	def __subclasscheck__(cls, subclass):
		'''issubclass(instance, class)'''
		pass#?


	def __lt__(self, obj):
		'''k<obj'''
		pass

	def __gt__(self, obj):
		'''k>obj'''
		pass

	def __eq__(self, obj):
		'''k==obj,默认是id(k)==id(obj)'''
		pass

	def __le__(self, obj):
		'''k<=obj'''
		pass

	def __ge__(self, obj):
		'''k>=obj'''
		pass

	def __ne__(self, obj):
		'''k!=obj'''
		pass




	def __add__(self, obj):
		'''k+obj'''
		pass

	def __sub__(self, obj):
		'''k-obj'''
		pass

	def __mul__(self, obj):
		'''k*obj'''
		pass

	def __matmul__(self, obj):
		'''k@obj, 矩阵乘法'''
		pass

	def __truediv__(self, obj):
		'''k//obj'''
		pass

	def __floordiv__(self, obj):
		'''k/obj'''
		pass

	def __mod__(self, obj):
		'''k%obj'''
		pass

	def __divmod__(self, obj):
		'''divmod(k,obj)'''
		pass

	def __pow__(self, obj):
		'''k**obj,或者pow(k,obj)'''
		pass

	def __lshift__(self, obj):
		'''k<<obj'''
		pass

	def __rshift__(self, obj):
		'''k>>obj'''
		pass

	def __and__(self, obj):
		'''k&obj'''
		pass

	def __or__(self, obj):
		'''k|obj'''
		pass

	def __xor__(self, obj):
		'''k^obj'''
		pass




	'''
	当上面左运算没找到时,会尝试找右运算
	'''

	def __radd__(self, obj):
		'''obj+k'''
		pass

	def __rsub__(self, obj):
		'''obj-k'''
		pass

	def __rmul__(self, obj):
		'''obj*k'''
		pass

	def __rmatmul__(self, obj):
		'''obj@k, 矩阵乘法'''
		pass

	def __rtruediv__(self, obj):
		'''obj//k'''
		pass

	def __rfloordiv__(self, obj):
		'''obj/k'''
		pass

	def __rmod__(self, obj):
		'''obj%k'''
		pass

	def __rdivmod__(self, obj):
		'''divmod(obj,k)'''
		pass

	def __rpow__(self, obj):
		'''obj**k,或者pow(obj,k)'''
		pass

	def __rlshift__(self, obj):
		'''obj<<k'''
		pass

	def __rrshift__(self, obj):
		'''obj>>k'''
		pass

	def __rand__(self, obj):
		'''obj&k'''
		pass

	def __ror__(self, obj):
		'''obj|k'''
		pass

	def __rxor__(self, obj):
		'''obj^k'''
		pass




	def __iadd__(self, obj):
		'''k+=obj'''
		pass

	def __isub__(self, obj):
		'''k-=obj'''
		pass

	def __imul__(self, obj):
		'''k*=obj'''
		pass

	def __imatmul__(self, obj):
		'''k@=obj, 矩阵乘法'''
		pass

	def __itruediv__(self, obj):
		'''k//=obj'''
		pass

	def __ifloordiv__(self, obj):
		'''k/=obj'''
		pass

	def __imod__(self, obj):
		'''k%=obj'''
		pass

	def __divmod__(self, obj):
		'''divmod(k,obj)'''
		pass

	def __ipow__(self, obj):
		'''k**=obj,或者pow(k,obj)'''
		pass

	def __ilshift__(self, obj):
		'''k<<=obj'''
		pass

	def __irshift__(self, obj):
		'''k>>=obj'''
		pass

	def __iand__(self, obj):
		'''k&=obj'''
		pass

	def __ior__(self, obj):
		'''k|=obj'''
		pass

	def __ixor__(self, obj):
		'''k^=obj'''
		pass




	def __neg__(self):
		'''-k'''
		pass

	def __pos__(self):
		'''+k'''
		pass

	def __abs__(self):
		'''abs(k)'''
		pass

	def __invert__(self):
		'''~k'''
		pass




	def __int__(self):
		'''int(k)'''
		pass

	def __complex__(self):
		'''complex(k)'''
		pass

	def __float__(self):
		'''float(k)'''
		pass

	def __bool__(self):
		'''bool(k), 没有这个方法时会尝试调用__len__判断是否非零'''
		pass

	def __round__(self):
		'''round(k)'''
		pass

	def __bytes__(self):
		'''bytes(obj)'''
		pass

	def __format__(self, format_spec):
		pass#?




	def __coerce__(self, num):
		pass#?

	def __prepare__(self):
		pass#?




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
__slots__#一个字符串的元组,用来限制该类的实例能拥有的属性
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


#特殊built-in函数
locals()#返回局部变量字典(str->obj)
globals()#返回全局变量字典(str->obj)
