# top
top
    -p xxx #筛选pid
    -u xxx #筛选用户
    -b #不断打印的模式
    -n 2 #指定打印次数，配合-b使用
    -d 3 #刷新时间

    q #退出
    c #显示完整执行命令
    M #按内存占用排序
    P #按CPU占用排序

# ps
ps
ps -aux #显示所有进程

# kill
kill xxx
kill -9 xxx #最高强度kill