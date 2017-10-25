#jh:func-base-similar,key-type

jupyter-notebook

#jupyter --generate-config
#c.NotebookApp.notebook_dir = u'D:newcode\python\jupyter'

#执行并跳到下一个cell
shift+enter
#执行,但不跳到下一个cell
ctrl+enter
#执行,并在下方创建新cell,并跳入
alt+enter

#进入当前cell的编辑模式
enter
#退出当前cell的编辑模式
esc

#删除当前cell
d d
#撤销删除
z

#显示行号
l

#跳到第一个cell
ctrl+home
#跳到最后一个cell
ctrl+end

#载入py文件
%load test.py
#运行py文件
%run test.py

#cell当成cmd用(加!)
!python test.py

#帮助
h

#切换到代码模式
y
#切换到markdown模式
m

#在上方插入新cell
a
#在下方插入新cell
b

#显示/隐藏输出内容
o

#打断kernel运行
i i
#重启kernel运行
o o
