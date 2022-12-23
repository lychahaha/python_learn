# netstat
netstat #显示所有网络连接和本地套接字
    -a #显示全部连接（包括监听中的）
    -n #显示ip和port的数字而不是名称
    -p #显示程序名

    --ip #只显示网络连接
    -x #只显示本地套接字
    -t #只显示tcp网络连接
    -u #只显示udp网络连接
    -l #只显示监听中的连接

netstat -i #显示网卡
netstat -s #显示连接统计信息
netstat -r #显示路由表

netstat -anp | grep 8080 #查看端口占用

# ifconfig
ifconfig #显示所有网卡信息

