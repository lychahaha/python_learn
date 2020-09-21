#jh:func-base-mixsimilar,key-type
import pdb

#停下来
pdb.set_trace()
#测试函数
pdb.runcall(fx, args)
#开始就停下来
python -m pdb xx.py
#交互模式出错后进入pdb调试模式
pdb.pm()

#帮助(help)
h
h l #查看某个命令的帮助


#继续运行(continue)
c

#下一行,不进入函数(next)
n

#下一行,进入函数(step)
s

#运行到函数返回(return)
r

#跳出循环(until)
unt #跳出当前堆栈/当前循环
unt 233 #执行到233行停下来

#跳到第x行(jump)
j 233

#退出程序(quit,exit)
q


#断点(break)
b #查看所有断点
b 233 #在xx行设置断点
b fx #在函数入口设置断点
b aaa/bbb.py:233 #在某文件设置断点
b 233,a==3 #设置条件,只有满足条件,才断下来
#一次性断点
tbreak 233

#设置条件(condition)
condition 4 a==3 #对第x个断点设置条件
condition 4 #取消条件

#清除断点(clear)
cl 3 #清除第x个断点
cl 3 5 7 #清除多个断点
cl #清除所有

#禁用/激活断点
disable 3 #禁用
disable 3 5 6 #禁用多个
enable 4 #激活
enable 4 6 8 #激活多个
ignore 3 4 #忽略第3个断点4次


#查看代码块(list)
l #查看当前代码,继续按会查看后面的代码
l . #始终查看当前运行到的位置
l 14 #查看14为中心的上下代码块
l 14,20 #查看14到20的代码块
#查看当前函数代码块(longlist)
ll

#查看参数(args)
a

#查看变量
p cnt
#更好看一点
pp cnt
#查看变量类型
whatis cnt

#执行命令(exec)
!x=233


#查看函数栈信息(主要为了看局部变量)
#查看所有函数栈(where,bt)
w
#跳到上一层函数栈(up)
u
u 3 #跳3层
#跳到下一层函数栈(down)
d
d 3


#快速输入命令
alias #查看所有别名和对应命令
alias name comd #给该命令命名(以后可以直接用这个name执行命令)
alias lk array[%1] #带参数的命令(用法是类似lk 3)
unalias name #取消这个命名


#重启
run
restart


#转换到交互模式
interact


'''
EOF,commands,debug,display,retval,rv,source,undisplay
'''