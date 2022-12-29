# 指定shell
#!/bin/bash


# 变量
## get
$xxx
"hehe${xxx}hehe" #字符串等复杂情况中获取
## set
xxx=xxxx #字符串赋值
xxx='xx xx' #带空格
xxx=$xxxx #变量赋值
xxx=`xxxx` #运行命令xxxx,结果赋值
## del
unset xxx
## 其他
readonly xxx #修改这个变量为常量


# 字符串
## set
xxx="xxxx"
xxx='xxx' #单引号内为raw，没有转义，没有变量
## 长度
${#xxx}
## 子串
${xxx:1:4} #两个数字表示起始坐标和长度
## 查找
`expr index "$string" i` #只能查找字符，这里查找i


# 数组
## get
${xxx[2]}
## set
xxx=(2 3 4)
xxx[1]=3
## 长度
${#xxx[@]}
## 迭代
${xxx[@]}
${xxx[*]} #所有元素拼接成一个字符串


# 关联数组（键值对）
## init
declare -A xxx
declare -A xxx=(['a']=1 ['b']=2)
## get
${xxx["a"]}
## set
xxx["a"]=1
## 长度
${#xxx[@]}
## 迭代
${xxx[@]} # 值
${!xxx[@]} # 键
${xxx[*]} # 所有值拼接成一个字符串


# 参数
$0 #脚本名字
$1 $2 ...#第x个参数
$* #所有参数拼接成一个字符串
$@ #所有参数拼接成一个字符串
$# #参数个数
$$ #当前pid
$? #返回值


# 数学运算
## 加减乘除
`expr $a + $b`
`expr $a - $b`
`expr $a \* $b`
`expr $a / $b`
`expr $a % $b`


# 输入输出
## read
read xxx
read xxx -p 'enter passwd:' #带上提示
## echo
echo $xxx
    -n #不换行
## printf
printf "%s %d %-4.2f\n" 2 3 "3.4"
%s %d %f # 字符串、整数、浮点数
%-4.2f #-表示左对齐（没有则右对齐），4表示长度，.2表示小数位数
## 重定向和管道
xxx > xxx.txt #输出到文件
xxx >> xxx.txt #追加输出
xxx < xxx.txt #从文件输入
xxx | xxxx #管道

# 退出
exit xxx #以这个退出码退出
$? #退出码

# if-else
## xxx的退出状态码是0才是true
if xxx
then
    xxxx
elif xxxx5
then
    xxxx6
else
    xxxx7
fi
## xxx成立就是true
if [ $var -eq 4 ]
then
    xxxx
fi
## 写成一行
if xxx; then xxx ; fi
## 条件
### 数值
[ $a -xxx $b ]
-eq #==
-ge #>=
-gt #>
-le #<=
-lt #<
-ne #!=
### 逻辑
[! xxx ] #非
[ $a -o $b ] #或
[ $a -a $b ] #与
if [[ xxx1  ||  xxx2 ]] #条件或
if [[ xxx1  &&  xxx2 ]] #条件与
### 字符串
[$a = $b ] #相等
[$a != $b ] #不等
[-z $b ] #长度是否=0
[-n $b ] #长度是否>0
### 文件
[ -xxx file ]
-d #是否存在并且是目录
-e #是否存在
-f #是否存在并且是文件
-r #是否存在并且可读
-s #是否存在并且非空
-w #是否存在并且可写
-x #是否存在并且可执行
-O #是否存在并且属当前用户
-G #是否存在并且当前用户属于这个组?
### 高级数学表达式,大于小于不需要转义
if (( $var1 ** 2 > 90 ))
### 高级字符串表达式(可以用正则表达式)
if [[ $user == r* ]]


# case
case $user in
    aaa) xxx
    ;;
    aab) xxx
    ;;
    *) xxx
    ;;
esac


# for
for var in  a b c
do
    xxx1
    xxx2
done
##一行写法
for var in a b c; do xxx1; xxx2; done;


# while
while cond
do
    xxx1
    xxx2
done
## 读入到停止写法
while read xxx #按ctrl+D停止


# 无限循环写法
while :
while true
for ((;;))


# break/continue
## 与C语言一样


# 函数
func(){
    echo $1 #获取参数
    echo $2
    echo ${10} #获取第10个参数以上时要加括号
    echo $# #获取参数个数
    return $bb
}

func 2 3 4
echo $? #获取返回值


# import
source xxx.sh