# 生成密钥
## id_rsa是私钥, id_rsa.pub是公钥
## 公钥给远程主机, 私钥在登录的时候用
ssh-keygen -t rsa -C "123@qq.com"

# 远程config
cat id_rsa.pub >> ~/.ssh/authorized_keys

# 本地config(用于快速ssh和vscode跳板等)
## ServerAliveInterval 60 表示每60秒心跳
## IdentityFile给的是私钥
## 局域网机器只比跳板机多了一行ProxyCommand

## ~/.ssh/config
Host jumpname
    HostName 192.168.1.1
    Port 22
    User username
    ServerAliveInterval 60
    ServerAliveCountMax 3
    TCPKeepAlive no
    StrictHostKeyChecking no
    IdentityFile C:\Users\lychahaha\.ssh\id_rsa
    PasswordAuthentication no
    IdentitiesOnly yes

Host gpuname
    HostName 192.168.1.2
    Port 22
    User username
    ServerAliveInterval 60
    ServerAliveCountMax 3
    TCPKeepAlive no
    StrictHostKeyChecking no
    IdentityFile C:\Users\lychahaha\.ssh\id_rsa
    PasswordAuthentication no
    IdentitiesOnly yes
    ProxyCommand C:\Windows\System32\OpenSSH\ssh.exe jumpname nc %h %p %r

Host gpuname_linux
    HostName 192.168.1.2
    Port 22
    User username
    IdentityFile ~/.ssh/id_rsa
    ProxyCommand ssh jumpname -W %h:%p


## 快速ssh
ssh jumpname

## 快速ssh(有跳板)
ssh -oProxyCommand='ssh -W %h:%p -i ~/.ssh/id_rsa -p 1234 -oStrictHostKeyChecking=no username@192.168.1.1' -p 1235 -oStrictHostKeyChecking=no -i ~/.ssh/id_rsa username@192.168.1.2



# git指定私钥
## 只能通过config指定(host,port)和私钥

## ~/.ssh/config
Host abc
    HostName abc.com
    Port 1234
    IdentityFile xxx/id_rsa

git clone github@abc.com:1234/def/ghi.git
git clone github@abc/def/ghi.git


# 设置ssh可用root远程登录
## 编辑/etc/ssh/sshd_config
PermitRootLogin yes
## 重启sshd服务
systemctl restart sshd


# SSH隧道
ssh -L 12349:127.0.0.1:34567 root@192.168.222.22