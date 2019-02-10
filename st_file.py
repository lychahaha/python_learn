#jh:func-base,key-type
import os,sys

#open
#mode默认为r
#mode有'r'(只读),'w'(只写),'a'(追加),'r+'/'w+'(读写),'a+'(追加+读写)
#'rb','wb','ab','rb+','wb+','ab+'为二进制的对应模式
f = open("123.txt", 'r')
f = open("123.txt", 'w')
#带编码读写,默认是平台的默认编码
f = open("123.txt", 'r', encoding='utf-8') 

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


import glob
#查找符合规则的文件名
glob.glob('D:/code/*.py') #list of str
'''
*:匹配0或多个字符
**:匹配文件,目录
?:匹配1个字符
[1-9]:匹配指定范围内的字符
[!1-9]:匹配不在范围内的字符
'''