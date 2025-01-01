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
tcpdump #linux里的wireshark（安卓用vnet）
    -i ens33 #指定网卡
    -w packets.pcap #指定保存文件（给wireshark用）
    -n #ip显示数字而不是名称
    -nn #ip和端口都显示数字而不是名称
    -XX #显示十六进制和ascii值的包信息（一般给wireshark用）
    -vvv #显示最全的信息（一般给wireshark用）
tcpdump host 123.45.67.89 #抓取特定主机
tcpdump src host 123.45.67.89 #指定发送源
tcpdump dst host 123.45.67.89 #指定目标源
tcpdump net 192.168.1.0/24 #抓取网段
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
nc -l 2345 #监听tcp端口
nc -l 2345 > a.txt #监听并将接收到的数据写入文件
    -l #监听(listen)
    -v #输出反馈(verbose)
    -u #udp模式
    -k #支持重复连接
    -4 #强制使用ipv4
    -6 #强制使用ipv6
    -p 12345 #指定客户端的端口

# iperf
## 下载
'''
win下载: https://iperf.fr/iperf-download.php
linux下载: yum install iperf
android下载: apk名为"Magic iPerf including iPerf3"
'''
## 服务端
iperf3 -s -B 12.34.56.78 -p 1234 #启动服务端，默认端口5201
## 客户端
iperf3 -c 12.34.56.78 -p 5201 #客户端访问服务器
## 其他选项
-t 60 #测试60秒，默认10秒
-I 10M #发送10M的数据包
-u #测试udp
-f M #指定bit单位


# ip
## addr(IP地址)
ip addr #查看IP
ip addr add 192.168.1.2/24 dev eth0 #增加ip地址（需指定网卡）
ip addr delete 192.168.1.2/24 dev eth0 #删除ip地址
## route(路由)
ip route #查看路由
ip route add 192.168.1.0/24 via 10.0.0.1 dev eth0 #增加路由（目标网络换成default则是添加默认路由）
ip route del 192.168.1.0/24 via 10.0.0.1 dev eth0 #删除路由
ip route del default dev eth0 #删除默认路由
ip route get 12.34.56.78 #查看该ip作为包的目标ip，这个包要怎么路由
### 路由表解释
192.168.1.0/24 dev eth0 src 192.168.1.3 #192.168.1.0/24表示目标网络/主机，dev eth0表示走哪个网卡，src 192.168.1.3表示本机以什么IP地址发包
default via 192.168.1.1 dev eth0 src 192.168.1.3 #default表示默认路由，via 192.168.1.1表示通过网关访问（无via则表明是局域网）
192.168.1.0/24 dev eth0 proto dhcp src 192.168.1.3 #proto dhcp表示是dhcp协议设的路由规则
192.168.1.0/24 dev eth0 scope link src 192.168.1.3 #scope link表示这个路由规则的作用范围为与指定接口连接的局域网
192.168.1.0/24 dev eth0 src 192.168.1.3 metric 20 #metric 20表示优先级（越小越优先）
## link(接口,网卡)
ip link #查看网卡
ip link set eth0 up #启动网卡（关闭是down）


# nmcli
# nmtui

# ipv4/ipv6
## ipv4
10.0.0.0/8 #LAN A类网络
172.16.0.0/12 #LAN B类网络
192.168.0.0/16 #LAN C类网络
127.0.0.0/8 #本地地址
224.0.0.0/8 #多播地址
169.254.0.0/16 #APIPA地址（没DHCP时操作系统自动分配的地址，不能用于上网）
255.255.255.255 #广播地址
## ipv6
### 地址结果：48+16+64:网络|子网|设备
### 访问http：http://[2400::1]:1234
### ipv6没有广播
::1 #本地地址
2000::/3 #互联网地址
fc00::/7 #局域网地址
fe80::/10 #仅交换机连接连通的地址(下称link-local)
ff02::1 #多播到link-local下的所有设备
ff02::2 #多播到link-local下的所有路由器
ff02::fb #多播到link-local下的所有mDNSv6服务器
ff02::1:2 #多播到link-local下的所有DHCPv6服务器
ff02::101 #多播到link-local下的所有NTP服务器
