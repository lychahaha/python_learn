# grep
grep pattern xxx #对xxx的每行进行匹配，输出匹配的行
    -v #输出不匹配的行
    -n #显示行号
    -c #输出多少行匹配
    -r #对文件夹里的所有文件进行筛选

# wc
wc xxx #统计xxx的行数，词数，字节数
    -l #只输出行数
    -w #只输出词数
    -c #只输出字节数
ls | wc -w #统计该目录有多少个文件

# sort
sort xxx #以行为单位,按字典序排序
    -n #按数字排序
    -M #按三字符的月份排序
    -r #倒序

# find
find root_dir -xxx xxx #查找文件
find ./ -name "*.c" #按名字查找
find ./ -ctime 20 #最近20天更新过的文件
find ./ -type f #查找所有“文件”