#jh:func-base,lazy-type
import re

# typical
pattern = r'(?P<name>[a-z1-9]+)@([a-z]+).([a-z]+).([a-z]+)'
string = 'My email is lijheng3@sysu.edu.cn, but i have a new one.'
ma=re.search(pattern, string)

ma.span() #(12,32)
ma.beg() #12
ma.end() #32
ma.group() #'lijheng3@sysu.edu.cn'
ma.groups() #('lijheng3', 'sysu', 'edu', 'cn')
ma.groupdict() #{'name': 'lijheng3'}
ma.expand(r'\g<name>@\g<2>.\3.\4') #'lijheng3@sysu.edu.cn'



# full

# ma:一个储存结果的类
ma.span() #[beg,end),匹配的区间下标
ma.span(ix) #匹配的第ix个分组的区间下标
ma.span(name) #匹配的分组名字为name的区间下标
ma.beg() #beg(同样可以加ix或name参数)
ma.end() #end

ma.group() #匹配的字符串
ma.group(ix) #匹配的第ix个分组的字符串
ma.group(name) #匹配的分组名字为name的字符串

ma.groups() #匹配的所有分组的字符串组成的元组
ma.groupdict() #匹配的所有带名字的分组组成的字典

ma.expand(pattern) #用匹配到的分组,填充到这个pattern里



# 查找
ma = re.search(r'bc', 'abcd')

# 从头匹配
ma = re.match(r'abc', 'abcd') #pattern必须是在string的开头


# 查找全部
# 找到的字符串相互是没有交集的
# 如果是有分组的,返回的是分组结果
ans = re.findall(r'bc', 'bcbcbc') #['bc','bc','bc']
ans = re.findall(r'c.c', 'cacbca') #['cac']
ans = re.findall(r'b(\d)c(\d)', 'b3c3 b2d1 b5c2') #[('3', '3'), ('5', '2')]
## 使用迭代器来获取所有的ma
## 有分组的情况下方便获取整个字符串或者获取匹配的位置
for ma in re.finditer(r'bc', 'bcbcbc')
    do_something(ma)

# 替换
ans = re.sub(r'\d+', '100', 'int a=1+2') #ans = 'int a=100+100' 
## 使用函数进行替换
func = lambda ma:str(int(ma.group())+1)
ans = re.sub(r'\d+', func, 'int a=1+2') #ans = 'int a=2+3' 
## 替换并统计替换次数
ans = re.subn(r'\d+', '100', 'int a=1+2') #ans = ('int a=100+100',2)


# 分割
ans = re.split(r':| ','immoc:C java python') #['immoc', 'C', 'java', 'python']




# 匹配数字/非数字
\d \D
# 匹配单词字符[a-zA-Z0-9]/非单词
\w \W
# 匹配空白/非空白
\s \S
# 匹配除\n的任何字符
.
# 匹配特定字符集/匹配除该字符集的字符
[dwk] [^dwk]
#re.search(r'\d+','The number is 24513') -> '24513'
#re.search(r'\w+','!hello123hello...').group() -> 'hello123hello'
#re.split(r'\s+','The number is 24513') -> ['The', 'number', 'is', '24513']
#re.findall(r'ca.e','cate care') -> ['cate', 'care']
#re.findall(r'[abd]\d', 'a3 b2 c5 d1') -> ['a3', 'b2', 'd1']

# 匹配任意次(包括0)
*
# 匹配至少一次
+
# 匹配0次或1次
?
# 匹配m次
{m}
# 匹配至少n次/匹配至多n次
{n,} {,n}
# 匹配m次到n次
{m,n}
#re.findall(r'ca\w*','cat ca carr carrr') -> ['cat', 'ca', 'carr', 'carrr']
#re.findall(r'ca\w+','cat ca carr carrr') -> ['cat', 'carr', 'carrr']
#re.findall(r'ca\w?','cat ca carr carrr') -> ['cat', 'ca', 'car', 'car']
#re.findall(r'ca\w{2}','cat ca carr carrr') -> ['carr', 'carr']
#re.findall(r'ca\w{2,3}','cat ca carr carrr') -> ['carr', 'carrr']

# 非贪婪模式
*? +? ??
#re.search(r'23\d+', '2333333').group() -> '2333333'
#re.search(r'23\d+?', '2333333').group() -> '233'


# 开头/结尾
^ $
#re.findall(r'^233','232 233') -> []
#re.findall(r'^232','232 233') -> ['232']

# 或逻辑
|
#re.findall(r'a\d|b\d','a3d bt5 c5d b3w ade bcb caq') -> ['a3', 'b3']
#re.findall(r'(?:a\d|b\d)d','a3d bt5 c5d b3w ade bcb caq') -> ['a3d']

# 作为一个分组(和限定范围)
()
# 作为某个名字的分组
(?P<name>)
# 引用第x个分组
\1 \2 ...
# 引用有名字的分组
(?P=name)
#re.search(r'(\w+)@([\.\w]+)', 'My email is lijheng3@sysu.edu.cn').groups() -> ('lijheng3', 'sysu.edu.cn')
#re.search(r'(?P<first>\w+)@(?P<second>[\.\w]+)', 'My email is lijheng3@sysu.edu.cn').groupdict() -> {'first': 'lijheng3', 'second': 'sysu.edu.cn'}
#[ma.group() for ma in re.finditer(r'(\d)=\1', '1=1 2=1 3=3')] -> ['1=1', '3=3']
#[ma.group() for ma in re.finditer(r'(?P<num>\d)=(?P=num)', '1=1 2=1 3=3')] ->['1=1', '3=3']

# 无捕获组(不作为分组,只用来限定范围)
(?:)
# 注释
(?#)
#re.findall(r'(a\d|b\d)d','a3d bt5 c5d b3w ade bcb caq') -> ['a3']
#re.findall(r'(?:a\d|b\d)d','a3d bt5 c5d b3w ade bcb caq') -> ['a3d']
#re.search(r'2(?#hahaha)33', '23333').group() -> '233'

# 匹配单词边界/非边界
\b \B
#re.findall(r'\w+er\b', 'number is not verb') -> ['number']
#re.findall(r'\w+er\B', 'number is not verb') -> ['ver']



# 零和非零开头的正整数
0|[1-9]\d*
# 浮点数
(-?\d+)(\.\d+)?
# 汉字
[\u4e00-\u9fa5]
