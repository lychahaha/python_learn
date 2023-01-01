# iptables

# firewall
systemctl status firewalld #查看是否打开防火墙

firewall-cmd --get-default-zone #默认区域
firewall-cmd --get-zones #列出当前所有区域
firewall-cmd --state #查看防火墙状态

firewall-cmd --runtime-to-permanent #将当前配置永久化

firewall-cmd --zone=public --add-port=8001/tcp #增加规则
    --permanent #重启依然生效
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