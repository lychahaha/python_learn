# iptables
## 四表五链
##              raw     mangle  nat     filter
## pre_routing  √       √       √       
## input                √               √
## forward              √               √
## output       √       √       √       √
## post_routing         √       √       

## 五链
##    ---------------------------------------------------
## -> | pre_routing -> ROUTE -> forward -> post_routing | ->
##    |                ↓                      ↑         |
##    |              input                  ROUTE       |
##    |                ↓                      ↑         |
##    |                ↓                    output      |
##    ---------------------------------------------------
##                     ↓                      ↑

## 查询规则
iptables -L #显示所有规则
## 每个规则包括以下列:
## target:动作
##   ACCEPT DROP REJECT
##   SNAT DNAT REDIRECT
##   LOG
##   跳到其他自定义链
## prot:协议
##   TCP UDP
## opt
## source:源地址
## destination:目标地址

## 增加自定义链
iptables -N mychain
## 跳转到自定义链
iptables -A INPUT -j BLOCK_1234

## 增加规则
iptables -I INPUT 2 -s 192.168.1.1/32 -j DROP
## -I INPUT 2   -I表示插入,INPUT为链名, 2为序号(同时表示优先级,若不指定则序号为1)
## -s 源地址
## -j 动作
iptables -A INPUT -t filter -p tcp -d 192.168.1.1/32 --sport 22 --dport 22,23 -j DROP
## -A 表示追加
## -t 表名(默认是filter)
## -p 协议
## -d 目标地址
## --sport 源端口(多个加逗号)
## --dport 目标端口

## 删除规则
iptables -D INPUT 2 #删除2号规则
iptables -D INPUT -s 192.168.1.1/32 #删除INPUT链中包括xxx源地址的所有规则
iptables -F #删除所有规则

## 保存配置
iptables-save > /root/xxx.bak

## 禁止端口访问
iptables -I INPUT -p tcp --dport 22 -j DROP

## 反向代理
iptables -t nat -A PREROUTING -p tcp --dport 1234 -j DNAT --to-destination 192.168.1.23:22
iptables -t nat -A POSTROUTING -p tcp --sport 22 -j SNAT --to-source 12.34.56.78:1234

## 蜜罐端口
iptables -A BLOCK_1234 -p tcp -m tcp --dport 1234 -m recent --set --name BLOCK_1234 --rsource
iptables -A BLOCK_1234 -p tcp -m recent --update --seconds 60 --hitcount 1 --name BLOCK_1234 --rsource -j DROP
iptables -A BLOCK_1234 -m recent --remove --name BLOCK_1234 --rsource
## 1.记录下访问1234/tcp的包，记录到BLOCK_1234表中，该表以源地址进行聚合
## 2.60秒内BLOCK_1234中有大于≥1条记录的，执行drop
## 3.自动清除BLOCK_1234超过60秒的记录
## 这些记录存储在/proc/net/xt_recent中

## 打log
iptables -I PREROUTING -t nat -j LOG --log-prefix "[NAT_PREROUTING_LOG]"





# firewall
systemctl status firewalld #查看是否打开防火墙

firewall-cmd --get-default-zone #默认区域
firewall-cmd --get-zones #列出当前所有区域
firewall-cmd --state #查看防火墙状态

firewall-cmd --runtime-to-permanent #将当前配置永久化

firewall-cmd --zone=public --add-port=8001/tcp #增加规则
    --permanent #重启生效（但现在不生效）
    --add-port=8001/tcp #添加端口规则
    --add-service=ssh #添加服务规则（本质上是添加端口，对应规则在/usr/lib/firewalld/services里）
    --add-source=192.168.1.100 #添加白名单ip(网段则写成如192.168.1.0/24)
    --add-interface=ens33 #添加网卡规则
    --add-masquerade #开启转发
    --add-forward-port=port=3306:proto=tcp:toaddr=192.168.1.100:toport=13306 #添加转发规则(源端口，协议，目标ip，目标端口)，如果转发本机则不要加目标ip
    --add-rich-rule="rule family=ipv4 source address='192.168.1.100' port protocol='tcp' port=3306 accept" #添加富规则(最后除了accept还有drop和reject)

firewall-cmd --zone=public --remove-port=8001/tcp #删除规则

firewall-cmd --zone=public --query-port=8001/tcp #查询规则
firewall-cmd --zone=public --query-masquerade #查询是否开启转发

firewall-cmd --reload #重启防火墙

firewall-cmd --list-all #列出当前区域所有规则
firewall-cmd --list-ports #列出当前区域所有端口规则
firewall-cmd --list-services #列出当前区域所有服务规则
firewall-cmd --list-sources #列出当前区域所有白名单ip
firewall-cmd --list-interfaces #列出当前区域所有网卡规则
firewall-cmd --list-forward-ports #列出当前区域所有转发规则

# SELinux
getenforce #查看selinux状态
setenforce 0 #设置selinux状态（0是permissive允许模式，1是enforcing强制模式）
## semanage,restorecon,getsebool,setsebool