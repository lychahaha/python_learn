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
	
try:
	pass
except NameError as e:
	print e

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

		
#名字错误
#名字未定义
NameError

#语法错误
SyntaxError

#IO错误
IOError

#除零错误
ZeroDivisionError

#值错误
#强制类型转换时
ValueError

#断言错误
#assert断言失败
AssertionError

#基础异常
BaseException

#异常
Exception

#键盘中断
KeyboardInterrupt
				