#ignore:[DNS],[DHCP],[email],[postfix],[dovecot],[mailx]

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


# nmcli

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
