nmap 192.168.10.5 #扫描主机所有端口

nmap -p 12345 192.168.2.2 #扫描主机的指定端口

nmap -sP 192.168.23.0/24 #扫描局域网ip

nmap -sT -O 192.168.23.0/24 #扫描局域网ip+猜操作系统
