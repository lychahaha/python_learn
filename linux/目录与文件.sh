# ignore: rmdir,nl,more

# ls
ls 
ls -l #详细列表
ls -a #隐藏文件
ls -R #遍历子孙目录
ls xx* #模糊匹配

# tree
tree
tree xxx #对xxx展开树目录
tree -L 2 #显示2层深度
tree -a #隐藏文件
tree -d #仅显示目录，不显示文件
tree -f #显示每个文件的相对路径
tree -p #显示文件权限
tree -s #显示文件大小
tree -h #human readable

# lsof
lsof xxx.txt #查看哪个进程占用该文件
lsof -p xxx #查看pid为xxx的进程占用了哪些文件
lsof -u xxx #查看xxx用户占用了哪些文件
lsof -i @192.168.1.13 #查看该ip对应套接字涉及的文件

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

# scp
scp xxx@192.168.2.2:/home/xx.txt /home/xx
    -P 6000 #指定端口
    -r #复制文件夹

# mv
mv src dst

# rm
rm xxx
rm -r xxx #删除目录
rm -f xxx #不提示

# ln
ln /xx/src /xx/dst #创建硬链接
    -s #创建软链接


# find
find root_dir -xxx xxx #查找文件
    -name "*.c" #按名字查找
    -user xx #按用户名查找
    -group xx #按组查找
    -ctime 20 #最近20天更新过的文件
    -type f #按类型查找（这里是查找所有“文件”）
    -size +50KB #按大小查找（这里是查找大于50KB的，减号则是小于）
    -perm 777 #按权限查找
    -exec ??? #查找后执行命令？

# locate
locate xxx.txt #快速查找文件（最好先执行updatedb建立索引）

# which
which xxx #快速查找二进制执行命令

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
