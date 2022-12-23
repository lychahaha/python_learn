# ignore: rmdir,nl,more

# ls
ls 
ls -l #详细列表
ls -a #隐藏文件
ls -R #遍历子孙目录
ls xx* #模糊匹配

# cd
cd xx
cd .. #返回上一级目录
cd - #返回上一次的目录
cd #返回home目录

# pwd
pwd #显示当前目录
pwd -P #用绝对路径显示当前目录

# mkdir
mkdir xx
mkdir xx/xx/xx -p #创建多级目录
mkdir xx -m 777 #设置特定权限

# cp
cp src_path dst_path
cp src_path dst_dir
cp -r src dst #复制目录

# mv
mv src dst

# rm
rm xxx
rm -r xxx #删除目录
rm -f xxx #不提示

# touch
touch xxx #创建新文件

# cat
cat xxx
    -n #加上行号
    -b #加上行号,但不包括空行

# tac
tac xxx #和cat相反，从最后一行开始输出文件

# less
less xxx
    # space/pagedown :往下一页
    # pageup :往上一页
    # /xxx :字符串搜索
    # ?xxx :字符串向上搜索
    # n :搜索下一个
    # N :搜索上一个
    # q :退出

# head
head xxx #查看前10行
head xxx -n 20 #自定义行数

# tail
tail xxx #查看后10行
tail xxx -n 20 #自定义行数
tail xxx -f #持续更新，直到ctrl+C结束

# file
file xxx #查看文件类型

# stat
stat xxx #查看文件属性
