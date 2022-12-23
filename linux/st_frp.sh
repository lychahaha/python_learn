#服务器和客户端共享同一份程序
#云服务器需要打开防火墙,允许端口7000和6000,5000进入

#bind_port(7000)用于客户端与服务器通信
#remote_port端口是用户ssh时用的端口,用于识别要连接哪个客户端
#local_ip和local_port是客户端给用户启动的ssh所写的ip和端口(所以连客户端只需要填127.0.0.1)

#服务器: frps frps.ini
#客户端: frpc frpc.ini


#服务器
nohup ./frps -c ./frps.ini &
#客户端
nohup ./frpc -c ./frpc.ini &
#外部机器
ssh -p 7000 jianheng@120.56.37.48


#frps.ini
[common]
bind_port = 7000 #监听端口

#frpc.ini
[common]
server_addr = 120.56.37.48   #公网服务器ip
server_port = 7000           #与服务端bind_port一致

[ssh]
type = tcp              #连接协议
local_ip = 127.0.0.1    #内网服务器ip
local_port = 22         #ssh默认端口号
remote_port = 6000      #自定义的访问内部ssh端口号

[ssh2]
type = tcp              #连接协议
local_ip = 172.18.23.23 #内网服务器ip
local_port = 12345      #ssh默认端口号
remote_port = 5000      #自定义的访问内部ssh端口号
