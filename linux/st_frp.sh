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
ssh -p 7000 xx@120.56.37.48


##################### 典型用法 #####################

#frps.ini
[common]
bind_port = 7000 #监听端口

#frpc.ini
[common]
server_addr = 120.56.37.48   #公网服务器ip
server_port = 7000           #与服务端bind_port一致

[ssh] #注意这个名字不能重复
type = tcp              #连接协议
local_ip = 127.0.0.1    #内网服务器ip
local_port = 22         #ssh默认端口号
remote_port = 6000      #自定义的访问内部ssh端口号



##################### 高级用法 #####################

# frps.ini
[common]
token = 123333 #frps验证frpc用的口令

# frpc.ini
[common]
token = 123333 #frps验证frpc用的口令(要和frps一致)
tls_enable = true #启用tls加密



##################### stcp通信 #####################
# stcp需要访客机提供口令才能通信，避免主人机被嗅探

# frps.ini
[common]
bind_port = 7000

# frpc.ini(主人机)
[common]
server_addr = 120.56.37.48
server_port = 7000

[ssh_server]
type = stcp
sk = abcdefg #主人机验证访客机用的口令
local_ip = 127.0.0.1
local_port = 22

# frpc.ini(访客机)
[common]
server_addr = 120.56.37.48
server_port = 7000

[ssh_visitor]
type = stcp
role = visitor
server_name = ssh_server #和主人机的代理名字对应
sk = abcdefg #主人机验证访客机用的口令
bind_addr = 127.0.0.1
bind_port = 6000

# ssh命令
## ssh_visitor[ssh->frpc(6000)] -> frps(7000) -> ssh_server[frpc->ssh(22)]
ssh -p 6000 xx@127.0.0.1



##################### xtcp通信 #####################
# 配置与stcp相似，但它是基于udp的p2p通信，因此也不能加密通信

# frps.ini
[common]
bind_port = 7000

# frpc.ini(主人机)
[common]
server_addr = 120.56.37.48
server_port = 7000

[ssh_server]
type = xtcp
sk = abcdefg #主人机验证访客机用的口令
local_ip = 127.0.0.1
local_port = 22

# frpc.ini(访客机)
[common]
server_addr = 120.56.37.48
server_port = 7000

[ssh_visitor]
type = xtcp
role = visitor
server_name = ssh_server #和主人机的代理名字对应
sk = abcdefg #主人机验证访客机用的口令
bind_addr = 127.0.0.1
bind_port = 6000

# ssh命令
## ssh_visitor[ssh->frpc(6000)] -> ssh_server[frpc->ssh(22)]
ssh -p 6000 xx@127.0.0.1
