#coding=utf-8
import string

a = 'strSTR123str'

#转大写
print a.upper()

#转小写
print a.lower()

#分割
print a.split()#这样相当于没有分割符
print a.split('t')#分割符,分割符会被删掉
print a.split('t', 1)#最多分割次数

#查找
print a.find('str')#返回头下标,没找到返回-1
print a.find('str', 3, 6)#在给定范围[beg,end)找
print a.index('str')#返回头下表,没找到抛出异常

#替换
print a.replace('123', '456')
print a.replace('str', 'haha', 1)#最多替换次数

#join
print '.'.join(a.split('t'))

#计数
print a.count('str')
print a.count('str', 3, 6)#在给定范围[beg,end)计数

#str转int,带进制转换
print string.atoi('FF',16)
print string.atoi('0xff',0)#基数为0时检查字符串前缀

#str转float
print string.atof('1.23e5')

#规范
b = a.capitalize()#首字母大写,后面小写

#编码解码
b = a.decode('utf-8')
b = a.encode('utf-8')

#判断字符种类
print a.isalnum()#全数字
print a.isalpha()#全字母
print a.islower()#全小写
print a.isupper()#全大写
print a.isspace()#全空白符

#居中
b = a.center(5,' ')#居中后字符串宽度,填充字符

#判断首尾
print a.startswith('xxx')
print a.endswith('xxx')

#去掉首尾符号
b = a.strip()#去掉首尾空白符
b = a.strip('k')#去掉首尾的k