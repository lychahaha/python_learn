#jh:func-base,lazy-type

import re

pa = re.compile(r'imooc')

#match(string[,pos[,endpos]])
#没匹配会返回空
ma = pa.match('imooc python')

ma.group()#返回匹配的字符串元组
ma.span()#匹配子串范围

ma = re.match(pattern,string)

ma.groups()#分组列表(数字索引)
ma.groupdict()#分组字典(名字索引)

#search(pattern, string, flags=0)
info = re.search(r'bc', 'abcd')

#findall(pattern, string, flags=0)
info = re.findall(r'bc', 'bcbcbc')

#sub(pattern, repl, string, count=0, flags=0)
#替换匹配的部分
#repl可以是字符串或是返回字符串的函数,参数是一个match
info = re.sub(r'\d+', '1001', 'int a = 1000')
info = re.sub(r'\d+', func, 'int a = 1000')

#split(pattern, string, maxsplit=0, flags=0)
#根据匹配分割
info = re.split(r':| ','immoc:C java python')

#匹配除\n的任何字符
.

#匹配字符集
[...]

#匹配数字/非数字
\d \D

#匹配空白/非空白
\s \S

#匹配单词字符[a-zA-Z0-9]/非单词
\w \W

#匹配任意次(包括0)
*

#匹配至少一次
+

#匹配0次或1次
?

#匹配m次
{m}

#匹配m次到n次
{m,n}

#非贪婪模式
*? +? ??

#开头
^

#结尾
$

#
\A \Z

#或逻辑
|

#作为一个分组
()

#第x个分组
\1 \2 ...

#给分组起名
(?P<name>)

#引用分组
(?P=name)