
#open(name[,mode[,buffer]])
#mode默认为r
#mode有'r'(只读),'w'(只写),'a'(追加),'r+'/'w+'(读写),'a+'(追加+读写)
#'rb','wb','ab','rb+','wb+','ab+'为二进制的对应模式
f = open("123.txt", 'r')
f = open("123.txt", 'w')

#read([size])
#默认读取全部
f.read()
f.read(10)

#readline([size])
#默认读一行,size是最大字节数
#\n会保留
f.readline()
f.readline(10)

#readlines([size])
#返回列表
#size为io.DEFAULT_BUFFER_SIZE的倍数
f.readlines()
f.readlines(1)

#write(str)
f.write('123')

#writelines(seq)
#不会自动补充\n
#seq是一个可迭代类型即可
f.writelines(seq)

#close()
f.close()

#iter(file)
#迭代器的一个元素是一行
iter_f = iter(f)

#flush()
#清空缓存(写)
f.flush()

#fileno()
#当前文件id
f.fileno()

#seek(offset[, whence])
#移动文件指针
#whence为参考系原点,可以选取os.SEEK_SET,os.SEEK_CUR,os.SEEK_END
f.seek(-5, os.SEEK_END)

#tell()
#当前文件指针
f.tell()

f.mode#打开模式
f.encoding#编码模式
f.closed#是否关闭

sys.stdin#标准输入
sys.stdout#标准输出
sys.stderr#标准错误

sys.argv#是个字符串的列表,第0个是程序名字

codecs.open('123.txt', 'w', 'utf-8')

os.open(filename, flag[, mode])
#flag包括os.O_CREAT(创建文件),os.O_RDONLY(只读),os.O_WRONLY(只写),os.RDWR(读写)
os.read(fd, buffersize)
os.write(fd, string)
os.lseek(fd, pos, how)
os.close(fd)

os.access(path, mode)#检查权限
#mode有os.F_OK,os.R_OK,os.W_OK,os.X_OK(读写)
os.listdir(path)#返回列表
os.remove(path)#文件删除
os.rename(old, new)#文件重命名
os.mkdir(path[, mode])#创建目录
os.makedirs(path[, mode])#创建多级目录
os.removedirs(path)#删除多级目录
os.rmdir(path)#删除空目录

os.path.exists(path)
os.path.isdir(s)
os.path.isfile(path)
os.path.getsize(filename)
os.path.dirname(p)#目录名
os.path.basename(p)#文件名

#处理ini文件
import ConfigParser

cfg = ConfigParser.ConfigParser()

cfg.read('125.ini')
cfg.sections()#返回section列表
cfg.items(section)#返回键值对(元组)列表
cfg.set(section, key, val)
cfg.remove_option(section, key)
cfg.remove_section(section)
cfg.add_section(section)
cfg.write(f)#f是一个写模式的文件对象