# 打印环境变量
printenv #显示所有全局环境变量
echo $xxx #显示某个环境变量
set #显示所有全局和局部环境变量

# 设置局部环境变量
xxx=abcde
xxx='ab cde'

# 把局部环境变量变成全局环境变量
export xxx

# 删除环境变量
unset xxx

# 数组环境变量
xxx=(a b c)
xxx[1]=d
echo ${xxx[1]}
echo $(xxx[*]) #打印所有元素

## 全局环境变量和局部环境变量的理解
## 这是针对shell来说的
## 局部是指只能在某个shell应用
## 全局是指能传到子shell去