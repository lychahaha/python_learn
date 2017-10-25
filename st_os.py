#jh:func-base-mixsimilar,key-type
import os
import shutil

os.name #os名字
os.environ #环境变量(一个dict)

os.getenv('PATH') #获取环境变量
os.cpu_count() #逻辑cpu数

os.sep #路径内目录分割符('\\')
os.extsep #包分割符('.')
os.pathsep #路径间分割符(';')
os.linesep #换行符('\r\n')

#当前目录
os.curdir #相对路径
os.getcwd() #绝对路径

os.rename(old, new) #文件重命名
os.remove(path) #文件删除
shutil.copy(src_path, dst_path) #文件复制
shutil.move(src_path, dst_path) #文件剪切
shutil.copytree(src, dst) #目录树复制
shutil.rmtree(path) #目录树删除

#列目录
os.listdir(path)
os.listdir('.')#获取当前目录所有目录和文件
#创建目录
os.mkdir(path)
os.makedirs(path) #多级
#删除目录
os.rmdir(path) #只能删除空目录
os.removedirs(path) #多级
#遍历子孙目录
##返回值是(path,list(dirs),list(files))
##topdown表示是自上而下还是自下而上
list(os.walk(top_path, topdown=True))

#路径并接
#(目录,目录或文件名)->路径
os.path.join(path, filename)
#路径分解
#路径->(目录,文件名+后缀)
os.path.split(path)
#扩展名分解
#路径->(目录+文件名,后缀)
os.path.splitext(path)

os.path.exists(path) #路径是否存在
os.path.isdir(path) #判断是否是目录
os.path.isfile(path) #判断是否是文件
os.access(path, mode) #检查权限
#mode有os.F_OK,os.R_OK,os.W_OK,os.X_OK(读写)

os.path.getsize(path) #文件大小
os.path.getmtime(path) #文件最后修改时间
os.path.getatime(path) #文件最后访问时间
os.path.getctime(path) #文件创建时间
os.path.dirname(path) #返回路径的目录
os.path.basename(path) #返回路径的文件名
os.path.abspath(path) #获取目录(文件)的绝对路径
os.path.abspath('.') #获取当前绝对路径

os.system("cls") #执行cmd命令

os.fork()
os.getpid() #获取进程id
os.getppid() #获取父进程id

os.abort() #报错,直接退出

#OS提供的文件读写API
os.open(filename, flag[, mode])
#flag包括os.O_CREAT(创建文件),os.O_RDONLY(只读),os.O_WRONLY(只写),os.RDWR(读写)
os.read(fd, buffersize)
os.write(fd, string)
os.lseek(fd, pos, how)
os.close(fd)
