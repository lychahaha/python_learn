# 开启转发
vim /etc/sysctl.conf #追加net.ipv4.ip_forward = 1
sysctl -p
# 开启nat
iptables -t nat -A POSTROUTING -s 10.8.0.0/24 -j MASQUERADE


# 安装easy-rsa生成证书和密钥
## 下载
yum install easy-rsa
apt install easy-rsa
## 配置（以下教程针对3.0版本）
mkdir /opt/easy-rsa
cp -r /usr/share/easy-rsa/* /opt/easy-rsa
cd /opt/easy-rsa
mv vars.example vars
vim vars #编辑证书用户信息
## 生成
./easyrsa init-pki
./easyrsa build-ca #生成根证书(注意要输入密码),pki/ca.crt
./easyrsa gen-req server nopass #生成服务器证书请求
./easyrsa sign server server #签发服务器证书,pki/issued/server.crt,pki/private/server.key
./easyrsa gen-req client nopass #生成客户端证书请求
./easyrsa sign client client #签发客户端证书,pki/issued/client.crt,pki/private/client.key
./easyrsa gen-dh #生成交换密钥,pki/dh.pem



# Linux openvpn服务端
## 下载
yum install openvpn
apt install openvpn
## 配置
mkdir /etc/openvpn/keys
cp ca.crt dh.pem server.crt server.key /etc/openvpn/keys #拷贝前面生成的证书和密钥
cp /usr/share/doc/openvpn/examples/sample-config-files/server.conf /etc/openvpn #拷贝示例配置文件
vim /etc/openvpn/server.conf #改协议(tcp/udp),改子网和端口,修改证书和密钥路径,加路由信息,修改对称加密算法为GCM
## server 10.0.1.0 255.255.255.0 子网为tun虚拟网卡的局域网，仅vpn服务器和客户端拥有
## push 172.16.30.102 255.255.255.0 表示该目标地址通过tun进行路由
## 启动
# openvpn --daemon --config /etc/openvpn/server.conf
systemctl start openvpn@server
systemctl enable openvpn@server



# Linux openvpn客户端
## 下载
yum install openvpn
apt install openvpn
## 配置
vim /etc/openvpn/client.conf #参考后面的配置文件
## 拷贝服务器生成的文件
cp ca.crt client.key client.crt /etc/openvpn
## 启动
systemctl start openvpn@client
systemctl enable openvpn@client



# Windows openvpn客户端
## 下载
## 配置
### 在OpenVPN/config中放入ca.crt client.key client.crt
### 在OpenVPN/config中创建client.ovpn配置文件,内容参考client.conf



# 加上密码认证
## 在server.conf加上
script-security 3                                   #允许使用自定义脚本
auth-user-pass-verify /etc/openvpn/check.sh via-env #指定认证脚本
username-as-common-name                             #用户密码登陆方式验证
## 创建/etc/openvpn/check.sh并加上权限,内容参考最后的check.sh
chmod +x /etc/openvpn/check.sh
## 创建密码表/etc/openvpn/openvpnfile
user1 passwd1
user2 passwd2
## 重启服务端
systemctl restart openvpn@server

## 在client.conf加上
auth-user-pass
## 重启客户端
systemctl restart openvpn@client



#server.conf
port 1194                                    #端口
proto udp                                    #协议
dev tun                                      #采用路由隧道模式
ca /opt/easy-rsa/pki/ca.crt                  #ca证书的位置
cert /opt/easy-rsa/pki/issued/server.crt     #服务端公钥的位置
key /opt/easy-rsa/pki/private/server.key     #服务端私钥的位置
dh /opt/easy-rsa/pki/dh.pem                  #证书校验算法  
server 10.8.0.0 255.255.255.0                #给客户端分配的地址池
push "route 172.16.1.0 255.255.255.0"        #允许客户端访问的内网网段
ifconfig-pool-persist ipp.txt                #地址池记录文件位置，未来让openvpn客户端固定ip地址使用的
keepalive 10 120                             #存活时间，10秒ping一次，120秒如果未收到响应则视为短线
max-clients 100                              #最多允许100个客户端连接
status openvpn-status.log                    #日志位置，记录openvpn状态
log /var/log/openvpn.log                     #openvpn日志记录位置
verb 3                                       #openvpn版本
client-to-client                             #允许客户端与客户端之间通信
persist-key                                  #通过keepalive检测超时后，重新启动VPN，不重新读取
persist-tun                                  #检测超时后，重新启动VPN，一直保持tun是linkup的，否则网络会先linkdown然后再linkup



# client.conf
client
dev tun
proto udp
remote 12.34.56.78 1194
resolv-retry infinite
nobind
ca ca.crt
cert client.crt
key client.key
verb 3
persist-key


# check.sh
#!/bin/bash
PASSFILE="/etc/openvpn/openvpnfile"   #密码文件 用户名 密码明文
LOG_FILE="/var/log/openvpn-password.log"  #用户登录情况的日志
TIME_STAMP=`date "+%Y-%m-%d %T"`
if [ ! -r "${PASSFILE}" ]; then
    echo "${TIME_STAMP}: Could not open password file \"${PASSFILE}\" for reading." >> ${LOG_FILE}
    exit 1
fi
CORRECT_PASSWORD=`awk '!/^;/&&!/^#/&&$1=="'${username}'"{print $2;exit}'    ${PASSFILE}`
if [ "${CORRECT_PASSWORD}" = "" ]; then
    echo "${TIME_STAMP}: User does not exist: username=\"${username}\",password=\"${password}\"." >> ${LOG_FILE}
    exit 1
fi
if [ "${password}" = "${CORRECT_PASSWORD}" ]; then
    echo "${TIME_STAMP}: Successful authentication: username=\"${username}\"." >> ${LOG_FILE}
    exit 0
fi
echo "${TIME_STAMP}: Incorrect password: username=\"${username}\", password=\"${password}\"." >> ${LOG_FILE}
exit 1