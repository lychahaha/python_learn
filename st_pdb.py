import pdb

#停下来
pdb.set_trace()

#测试函数
pdb.runcall(fx, args)


#帮助(help)
h

#运行的代码块(list)
l

#断点(break)
#查看所有断点
b 
#在xx行设置断点
b 233
#在函数入口设置断点
b fx

#条件断点
#对第x个断点,只有满足条件,才断下来
condition 4 a==3

#清除断点
#清除第x个断点
cl 3
#清除所有
cl

#禁用/激活断点
#禁用
disable 3
#激活
enable 4

#继续运行(continue)
c

#下一行,不进入函数(next)
n

#下一行,进入函数(step)
s

#运行到函数返回(return)
r

#跳(jump)
j 233

#查看参数(args)
a

#查看变量
p cnt

#修改变量
!x=233

#退出程序(quit)
q

#查看函数栈信息
#查看所有函数栈
w
#查看上一层函数栈
u
#查看下一层函数栈
d
