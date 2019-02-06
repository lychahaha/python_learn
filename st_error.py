#jh:mix-base,key-type
import traceback
import os,sys

raise MyError('No files')

assert a==1, 'a must be 1'

exc_type, exc_value, exc_traceback = sys.exc_info()#第三个没用
exc_traceback = traceback.format_exc()

#成功:try
#失败:try->except
try:
	pass
except:
	pass

#指定错误类型
try:
	pass
except NameError as e:
	print(e)
except IOError as e:
	print(e)
except (AttributeError,AssertionError) as e:
	print(e)

#继续往上报错
try:
	pass
except:
	print(e)
	raise

#成功:try->else
#失败:try->except
try:
	pass
except:
	pass
else:
	pass

#成功:try->finally
#失败:try->except->finally
try:
	pass
except:
	pass
finally:
	pass

#成功:try->else->finally
#失败:try->except->finally
try:
	pass
except:
	pass
else:
	pass
finally:
	pass


with expression as val:
	#__enter__()
	pass
	#__exit__()

def __enter__(self):
	pass
	return self#不一定是self

def __exit__(self, exc_type, exc_value, traceback):
	pass


NameError #名字错误,名字未定义
SyntaxError #语法错误
IOError #IO错误
ZeroDivisionError #除零错误
ValueError #值错误,强制类型转换时
AssertionError #断言错误
KeyboardInterrupt #键盘中断
BaseException #基础异常
Exception #异常

