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
    -f #忽略大小写
    -u #去重
    -t #指定key间隔符
    -k #指定第几列是key（搭配-t使用）

# tr
echo hehe | tr a-z A-Z #逐字符替换（这里是转换大小写）

# cut
cut -x xxx.txt #按列取值
    -b 3 #取每行的第x个字节
    -d : #自定义分割符（这里是冒号）
    -f 2 #取第几列

# diff
diff a.txt b.txt
    -y #并排显示
    -q #仅显示是否相同（相同是不会打印任何信息）

# sed

# awk
