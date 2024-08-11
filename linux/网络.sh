#ignore:[DNS],[DHCP],[email],[postfix],[dovecot],[mailx]

# 设置ip转发
net.ipv4.ip_forward=1 #在/etc/sysctl.conf里加入这句

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
## 显示所有网卡信息
ifconfig 
## 虚拟网卡
ifconfig eth0:0 192.168.2.2 netmask 255.255.255.0 #增加虚拟网卡,注意网卡的名字是物理网卡加:0的格式
ifconfig eth0:0 up #启动虚拟网卡
ifconfig eth0:0 down #关闭虚拟网卡
ifconfig eth0:0 del #删除虚拟网卡


# tcpdump
tcpdump #linux里的wireshark
    -i ens33 #指定网卡
    -w packets.pcap #指定保存文件（给wireshark用）
    -n #ip显示数字而不是名称
    -nn #ip和端口都显示数字而不是名称
    -XX #显示十六进制和ascii值的包信息（一般给wireshark用）
    -vvv #显示最全的信息（一般给wireshark用）
tcpdump host 123.45.67.89 #抓取特定主机
tcpdump src host 123.45.67.89 #指定发送源
tcpdump dst host 123.45.67.89 #指定目标源
tcpdump port 8080 #指定端口
tcpdump tcp #指定协议

tcpdump tcp[20:2]=0x4745 or tcp[20:2]=0x4854 #抓取http包

# ping
ping 123.45.67.89
    -c 4 #指定次数

# tracepath
tracepath 123.45.67.89

# nslookup
nslookup baidu.com # DNS查找

# nc
##Linux: apt install netcat
##windows: https://eternallybored.org/misc/netcat/
nc 12.34.56.78 2345 #创建tcp连接（后续可输入任意数据进行传输）
nc 12.34.56.78 2345 < a.txt #传输文件
nc -lv 2345 #监听tcp端口
nc -lv 2345 > a.txt #监听并将接收到的数据写入文件
    -l #监听(listen)
    -v #输出反馈(verbose)
    -u #udp模式
    -k #支持重复连接
    -4 #强制使用ipv4
    -6 #强制使用ipv6

# iperf
## 下载
'''
win下载: https://iperf.fr/iperf-download.php
linux下载: yum install iperf
android下载: apk名为"Magic iPerf including iPerf3"
'''
## 服务端
iperf3 -s -p1234 #启动服务端，默认端口5201
## 客户端
iperf3 -c 12.34.56.78 -p 5201 #客户端访问服务器
## 其他选项
-t 60 #测试60秒，默认10秒
-I 10M #发送10M的数据包
-u #测试udp
-f M #指定bit单位


# nmcli
# nmtui

# route
route -n #显示路由表
## 路由表中的flag含义
## U:活动路由，基本都带
## G:发给网关，不给该标志意味着发本地处理
## H:目标是一台主机

route add -host 123.45.67.89 dev eth0 #添加路由规则
route del -host 123.45.67.89 dev eth0 #删除路由规则
ip route add default gw 192.168.0.1 dev eth0 #添加默认路由
ip route del default dev eth0 #删除默认路由
## 参数含义
## -host:目标是主机
## -net:目标是网络，用192.168.0.0/24表示，或只有ip，通过netmask 255.255.255.0表示子网掩码
## default:表示默认路由
## gw:网关地址
## dev:网卡
