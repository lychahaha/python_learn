#coding=utf-8

#format
s = "{}+{}={}"
s.format(2,3,2+3)

s = "{1}+{0}={2}" #设置参数顺序
s = "{0:4}" #控制宽度(前面是参数顺序)
s = "{:.3}" #小数位数
s = "{:<7.3}" #居左,宽度7,小数位数3(>是居右,^是居中)


a = 'strSTR123str'
#合并
print '.'.join(a.split('t'))
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
#计数
print a.count('str')
print a.count('str', 3, 6)#在给定范围[beg,end)计数

#转大小写
print a.upper()
print a.lower()
print a.casefold()#所有语言的小写
print a.swapcase()#所有语言的反转

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

#判断字母
print a.isalpha()#全字母(不止英文)
print a.islower()#全小写
print a.isupper()#全大写
print a.isspace()#全空白符
#判断数字
print a.isdecimal()#判断普通数字,全角数字
print a.isdigit()#判断普通数字,全角数字,(byte数字),罗马数字
print a.isnumeric()#判断普通数字,全角数字,罗马数字,汉字数字
print a.isalnum()#isdecimal||isdigit||isnumeric
#其他判断
print a.isprintable()#能否打印

#align
b = a.center(5,' ')#居中后字符串宽度,填充字符
b = a.ljust(5, ' ')#居左后字符串宽度,填充字符
b = a.rjust(5, ' ')#居右后字符串宽度,填充字符

#tab->space
print a.expandtabs(tabsize=8)

#判断首尾
print a.startswith('xxx')
print a.endswith('xxx')

#去掉首尾符号
b = a.strip()#去掉首尾空白符
b = a.strip('k')#去掉首尾的k
b = a.lstrip()#去掉首
b = a.rstrip()#去掉尾
