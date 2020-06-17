#cd
    cd xx
    cd #返回home目录
    cd .. #返回上一级的目录
    cd - #返回上一次的目录

#ls
    ls
        -a #显示隐藏文件
        -F #区分目录和文件
        -R #遍历子目录
        -l #显示详细信息
    '''
    -rw-r--r-- 1 ljhandlwt 197609  282 7月   8 08:57  desktop.ini
    文件类型
    '''

    #模糊匹配
    ls xxx?
    ls xxx*

#创建
    touch a.txt #创建新文件
    mkdir xxx_dir

#复制
    cp src_path dst_path
    cp src_path dst_dir
    cp -r src dst #复制文件夹
#移动
    mv src_path dst_path
    mv src_path dst_dir
    mv src_dir dst_dir #可以直接移动文件夹
#删除
    rm xxx
        -r #递归删除
        -f #强制删除不提示

    rm -rf xxx

#查看文件信息
    stat xxx #查看文件信息

    file xxx #查看文件类型
#查看文本内容
    cat xxx #查看文本内容
        -n #加上行号
        -b #加上行号,但不包括空行
        -s #多个空行变成一个空行
        -T #把制表符变成^I

    more
    less

    tail xxx #查看文本末尾10行
        -n xxx #查看末尾xxx行
        -f #保持活动状态,文本有更新时它也会更新

#进程
    ps #显示进程信息

    top
        -p xxx #显示pid为xxx的进程信息
        -u xxx #显示user为xxx的进程信息
        i 切换是否显示空闲进程
        q 退出

    kill xxx

#硬盘
    df #查看磁盘分区使用情况(挂载情况)
        -h #动态地以KMGT等为单位

    du #显示当前目录的所有子目录的存储占用
        -h #动态地以KMGT等为单位

    du -sh #查看当前目录的存储占用
    
    mount #显示当前挂载的设备列表

    fdisk -l #查看物理硬盘情况

    mount /dev/sdc1 /aaa #将分区挂载到某目录下(这个目录需要原本存在)
    umount /dev/sdc1 #通过硬盘分区卸载
    umount /aaa #也可以通过挂载目录卸载

    mkfs.ext4 /dev/sdc1 #格式化分区

#数据处理
    sort xxx #以行为单位,按字典序排序
        -n #按数字排序
        -M #按三字符的月份排序
        -r #倒序

    grep pattern xxx #对xxx的每行进行匹配
        -v #输出不匹配的行
        -n #显示行号
        -c #输出多少行匹配

    wc "aa bb cc" #统计字符串的行数,词数,字节数
        -l #只输出行数
        -w #只输出词数
        -c #只输出字节数
    ##样例
    ls | wc -w #统计该目录有多少个文件

#压缩
    zip -r xxx.zip xxx_dir #把xxx_dir目录压缩成xxx.zip
    unzip xxx

    tar
        -z #输出重定向给gzip
        -c #创建新tar文件
        -x #从tar文件提取文件
        -v #显示提取文件名
        -f xxx #输出到xxx或从xxx提取

    tar -cvf xxx.tar xxx_dir #压缩
    tar -xvf xxx.tar #解压
    tar -zxvf xxx.tgz #解压tgz文件

#环境变量
    printenv #显示所有全局环境变量
    echo $xxx #显示某个环境变量
    set #显示所有全局和局部环境变量

    #设置局部环境变量
    xxx=abcde
    xxx='ab cde'
    #把局部环境变量变成全局环境变量
    export xxx
    #删除环境变量
    unset xxx
    #数组环境变量
    xxx=(a b c)
    xxx[1]=d
    echo ${xxx[1]}
    echo $(xxx[*]) #打印所有元素

#命令命名
    alias -p #显示命令别名
    alias li='ls -il' #设置命令别名

#权限
    chmod code xxx #改变xxx的权限为code
    chmod [ugoa] [+-=] [rwxXstugo] #改变权限
        u #用户
        g #组
        o #其他
        a #全部
        + #加上权限
        - #减去权限
        = #权限赋值

#计算器
    bc #计算器


#sh
    #打印
    echo xxx
        -n #不换行

    #赋值
    xxx=xxxx #字符串赋值
    xxx='xx xx' #带空格
    xxx=$xxxx #变量赋值
    xxx=`xxxx` #运行命令xxxx,结果赋值

    #stdio
    xxx > xxx.txt #输出到文件
    xxx >> xxx.txt #追加输出
    xxx < xxx.txt #从文件输入
    xxx | xxxx #管道

    #退出
    exit xxx #以这个退出码退出
    $? #退出码

    #if-else
    # xxx的退出状态码是0才是true
    if xxx
    then
        xxxx
    elif xxxx5
    then
        xxxx6
    else
        xxxx7
    fi

    # xxx成立就是true
    if [ $var -eq 4 ]
    then
        xxxx
    fi

    #数值
    n1 -xxx n2
    -eq #==
    -ge #>=
    -gt #>
    -le #<=
    -lt #<
    -ne #!=
    #字符串
    str1 = str2
    str1 != str2
    str1 < str2 #小于号要转义,大于号也是
    str1 > str2
    -n str1 #长度是否>0
    -z str1 #长度是否=0
    #文件
    -xxx file
    -d #是否存在并且是目录
    -e #是否存在
    -f #是否存在并且是文件
    -r #是否存在并且可读
    -s #是否存在并且非空
    -w #是否存在并且可写
    -x #是否存在并且可执行
    -O #是否存在并且属当前用户
    -G #是否存在并且当前用户属于这个组?
    file1 -nt file2 #是否更新
    file1 -ot file2 #是否更旧

    if [ xxx1 ] && [ xxx2 ]
    if [ xxx1 ] || [ xxx2 ]

    # 高级数学表达式,大于小于不需要转义
    if (( $var1 ** 2 > 90 ))
    # 高级字符串表达式(可以用正则表达式)
    if [[ $user == r* ]]

    #case
    case $user in
    aaa | aab )
        xxx
    aac )
        xxx2
    * )
        xxx0
    esac

