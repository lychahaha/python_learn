#coding=utf-8

import os

#os名字
os.name
#环境变量(一个dict)
os.environ
#获取环境变量
os.getenv('PATH')

#获取目录(文件)的绝对路径
os.path.abspath(path)
os.path.abspath('.')#获取当前绝对路径

#路径并接
#(目录,目录或文件名)->路径
os.path.join(path, filename)
#路径分解
#路径->(目录,文件名+后缀)
os.path.split(path)
#扩展名分解
#路径->(目录+文件名,后缀)
os.path.splitext(path)

#文件重命名
os.rename(old, new)
#文件删除
os.remove(path)

#列目录
os.listdir(path)
os.listdir('.')#获取当前目录所有目录和文件

#路径是否存在
os.path.exists(path)
#判断是否是目录
os.path.isdir(path)
#判断是否是文件
os.path.isfile(path)

#文件大小 
os.path.getsize(path)
#返回路径的目录 
os.path.dirname(path)
#返回路径的文件名 
os.path.basename(path) 

#创建目录
os.mkdir(path)
#删除目录
os.rmdir(path)

#fork
os.fork()
#获取进程id
os.getpid()
#获取父进程id 
os.getppid()