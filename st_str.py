#jh:func-base-mixsimilar,full-type

a = 'strSTR123str'

#合并
'.'.join(a.split('t'))
#分割
a.split()#这样相当于没有分割符
a.split('t')#分割符,分割符会被删掉
a.split('t', 1)#最多分割次数
a.splitlines()#相当于a.split('\n')

#查找
a.find('str')#返回头下标,没找到返回-1
a.find('str', 3, 6)#在给定范围[beg,end)找
a.index('str')#返回头下表,没找到抛出异常
a.partition('123')#原字符串拆成三部分返回,(beg,target,end),如果没找到则返回(src,'','')
#替换
a.replace('123', '456')
a.replace('str', 'haha', 1)#最多替换次数
#计数
a.count('str')
a.count('str', 3, 6)#在给定范围[beg,end)计数

#format
s = "{}+{}={}"
s.format(2,3,2+3)
##例子
s = "{1}+{0}={2}" #设置参数顺序
s = "{a}+{b}={c}" #使用名字索引, s.format(a=1,b=2,c=3)
s = "{0:4}" #控制宽度(前面是参数顺序)
s = "{:.3}" #小数位数
s = "{:<7.3}" #居左,宽度7,小数位数3(>是居右,^是居中)

#去掉首尾符号
b = a.strip()#去掉首尾空白符
b = a.strip('k')#去掉首尾的k
b = a.lstrip()#去掉首
b = a.rstrip()#去掉尾

#转大小写
a.upper()
a.lower()
a.casefold()#所有语言的小写
a.swapcase()#所有语言的反转
b = a.capitalize()#首字母大写,后面小写

#判断首尾
a.startswith('xxx')
a.endswith('xxx')
#判断字母
a.isalpha()#全字母(不止英文)
a.islower()#全小写
a.isupper()#全大写
a.isspace()#全空白符
#判断数字
a.isdecimal()#判断普通数字,全角数字
a.isdigit()#判断普通数字,全角数字,(byte数字),罗马数字
a.isnumeric()#判断普通数字,全角数字,罗马数字,汉字数字
a.isalnum()#isdecimal||isdigit||isnumeric
#其他判断
a.isprintable()#能否打印

#align
b = a.center(5,' ')#居中后字符串宽度,填充字符
b = a.ljust(5, ' ')#居左后字符串宽度,填充字符
b = a.rjust(5, ' ')#居右后字符串宽度,填充字符
b = a.zfill(10)#左对齐,前面补零(针对数字)

#tab->space
a.expandtabs(tabsize=8)

#格式转换
#char<->int
x = ord('3')
s = chr(51)
#16进制<->int
x = int('0x3', base=16)
s = hex(10)
#byte<->str
s = b.decode('utf-8')
b = s.encode('utf-8')
