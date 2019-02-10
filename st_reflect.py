import os,sys
import inspect

type(obj)
id(obj)
dir(obj)

#函数栈信息
fs = inspect.stack() #list
f = fs[0] #cur func
f[0] #f.frame,frame对象
f[1] #f.filename
f[2] #f.lineno
f[3] #f.function,函数名字
f[4] #f.code_context,当前行的代码
f[5] #f.index?

#静态
s = inspect.getsource(obj) #源代码
s = inspect.getabsfile(obj) #所属文件的绝对路径