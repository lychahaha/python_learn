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
## 最后一列:大部分规则细节都体现在该列
iptables -nvL #一般更常用这个
## -n:使用数字代替ip和端口的名字
## -v:显示更详细的信息(包括以下额外的列)
## pkgs:匹配该规则的数据包数量
## bytes:匹配该规则的数据包字节数
## in:源网卡
## out:目标网卡
iptables -nvL INPUT #指定对应链
iptables -nvL -t nat #指定表(默认是filter)


## 自定义链
iptables -N mychain #创建自定义链
iptables -X mychain #删除自定义链

## 默认策略
iptables -P INPUT DROP #只能是ACCEPT或DROP


## 增加规则
iptables -A INPUT -p tcp -d 192.168.1.0/24 --dport 22 -j DROP

-I INPUT 2 #表示插入到INPUT链的第2个规则
-R INPUT 2 #表示修改INPUT链的第2个规则
-A INPUT #表示追加到INPUT链的最后面
-t nat #表名(raw|mangle|filter|nat|security,默认是filter)
-p tcp #协议(tcp|udp|icmp...)
-s 12.34.56.78 #源地址
-d 12.34.56.78 #目标地址
--sport 22 #源端口
--dport 22,23 #目标端口(多个用逗号隔开)
-j ACCEPT #动作(ACCEPT|REJECT|DROP,SNAT|DNAT|REDIRECT|MASQUERAED,TRACE,LOG,自定义链|RETURN,SET)
-m xx #使用不同模块的功能

--icmp-type time-exceeded #指定icmp包的类型

-j LOG --log-prefix "[UFW block]" #指定使用哪个log进行记录
-j LOG --log-level 3 #指定日志等级(默认是warning, 0:emerg 1:alert 2:crit 3:err 4:warning 5:notice 6:info 7:debug)
-j MASQUERAED --random #从端口受限锥形NAT切换至对称型NAT
-j MASQUERAED --to-ports 30000-40000 #指定用来映射的端口（需要在前面指定协议）
-j SET --add-set set_x src #将源ip添加到指定集合里
-j SET --add-set set_x src --exist #如果源ip就在指定集合中,会刷新时间

-m conntrack
    --ctstate NEW #tcp连接的状态(NEW|RELATED|ESTABLISHED|INVALID)
-m recent
    --set #记录
    --update --seconds 30 --hitcount 6 #检查是否在x时间内记录超过y次
    --name xxx #指定记录用的记录本别名,一般配合--set和--update使用，记录存储在/proc/net/xt_recent中
-m limit
    --limit 3/min
    --limit-burst 10
-m addrtype
    --dst-type BROADCAST #指定目标地址类型(LOCAL|MULTICAST|BROADCAST)
-m set
    --match-set set_x src #判断源ip是否在指定集合里
    ! --match-set set_x src #判断源ip是否不再指定集合里

## 删除规则
iptables -D INPUT 2 #删除2号规则
iptables -D INPUT -s 192.168.1.1/32 #删除INPUT链中包括xxx源地址的所有规则
iptables -F #删除所有规则

## 连接情况
apt install conntrack
conntrack -L

## 案例
### 禁止端口访问
iptables -I INPUT -p tcp --dport 22 -j DROP

### 出口上网(正向代理)
iptables -t nat -A POSTROUTING -s 192.168.1.0/24 -j MASQUERAED

### 端口映射(反向代理)
iptables -t nat -A PREROUTING -p tcp -d 12.34.56.78 --dport 8022 -j DNAT --to-destination 192.168.1.5:22

### 代理(双向代理)
iptables -t nat -A PREROUTING -p tcp -d 12.34.56.78 --dport 8022 -j DNAT --to-destination 192.168.1.5:22
iptables -t nat -A POSTROUTING -p tcp -d 192.168.1.5 --dport 22 -j MASQUERAED

### 陷阱端口
ipset create ban_ip hash:ip timeout 3600 -exist #设置ipset,3600秒才取消黑名单,可刷新
iptables -I INPUT 1 -p tcp --dport 22 -j BAN #掉到陷阱
iptables -A BAN -j SET --add-set ban_ip src
iptables -A BAN -j DROP

### 转发过滤
iptables -A FORWARD -p tcp -d 192.168.1.5 --dport 22 -j SSH
iptables -A SSH -j ACCEPT #先加其他过滤规则，最后才加这个

### 转发过滤(区域限制)
ipset create zone_ip hash:ip #设置ipset
ipset add zone_ip 12.34.56.78 #增加ip
iptables -A SSH -m set ! --match-set zone_ip -j DROP

### 转发过滤(端口敲门)
ipset create ban_ip hash:ip timeout 30 -exist #设置ipset,30秒过期,可刷新
iptables -A SSH -m set ! --match-set knock_ip -j DROP #没敲门
iptables -I INPUT 1 -p tcp -d 12.34.56.78 --dport 12345 -j KNOCK #敲门处理
iptables -A KNOCK -j SET --add-set knock_ip src
iptables -A KNOCK -j DROP

### 转发过滤(防爆破)
ipset create ban_ip hash:ip timeout 3600 -exist #设置ipset,3600秒才取消黑名单,可刷新
iptables -A SSH -m set --match-set ban_ip -j SSH_BAN #已进黑名单的更新黑名单
iptables -A SSH -m recent --set #记录
iptables -A SSH -m recent --update --seconds 30 --hitcount 3 -j SSH_BAN #规定时间内达到次数则进黑名单
iptables -A SSH_BAN -j SET --add-set ban_ip src --exist
iptables -A SSH_BAN -j DROP

### 跟踪
modprobe xt_TRACE #打开trace功能
echo nf_log_ipv4 > /proc/sys/net/netfilter/nf_log/2 #设置trace的log目录
iptables -t raw -A PREROUTING -s 192.168.1.2 -j TRACE #跟踪192.168.1.2发来的包在防火墙里经过哪些链
cat /var/log/kern.log | grep TRACE #查看log

### 打log
iptables -A INPUT -s 192.168.1.2 -j LOG --log-prefix "[hehe]"
cat /var/log/kern.log | grep hehe #查看log


## 保存配置
iptables-save > /root/xxx.bak



# ipset
## 用于辅助iptables

## 安装
apt install ipset

## 创建集合
ipset create set_a hash:ip #hash为存储形式(还有bitmap|list)，ip为数据类型(还有net|mac|port|iface)
ipset create set_a hash:ip timeout 200 #设置元素的默认存在时间为200秒
ipset create set_a hash:ip -exist #如果该集合已存在，这样写不会报错
## 删除集合
ipset destroy set_a
## 列出所有集合
ipset list
## 保存集合
ipset save set_a -file a.txt
## 载入集合
ipset restore -file a.txt
## 清空集合
ipset flush set_a

## 增加元素
ipset add set_a 12.34.56.78
ipset add set_a 12.34.56.78 timeout 200 #设置元素的存在时间(如果设为0则是永久)
ipset add set_a 12.34.56.78 -exist #覆盖增加，将刷新存在时间
## 删除元素
ipset del set_a 12.34.56.78
## 判断元素是否在集合里
ipset test set_a 12.34.56.78





# ufw

##启用|禁用防火墙
ufw enable #启用
ufw disable #禁用

## 查看防火墙状态
ufw status #基础查看
ufw status numbered #给规则加上序号(方便增删)
ufw status verbose #详细查看状态

## 开启关闭日志
ufw logging on #打开
ufw logging off #关闭 

## 设置默认规则
## ufw default allow|deny|reject incoming|outgoing|routed
ufw default deny incoming #默认拒绝所有入站流量
ufw default allow outgoing #默认允许所有出站流量

## 根据端口设置规则
## ufw [insert NUM] allow|deny|reject|limit [in|out] [log|log-all] PORT[/PROTO]
## [in|out]不指代的话，默认是in(入站流量)，
ufw allow 22 #允许22端口
ufw deny 80/tcp #拒绝80端口的tcp流量
ufw limit 22 #限速22端口(30秒内最多6次)
ufw deny out 443 #拒绝443端口的主动出站流量
ufw insert 2 allow 80 #该规则插入作为第二条

## 根据源IP和目的IP设置规则
## ufw [insert NUM] allow|deny|reject|limit [in|out [on INTERFACE]] [log|log-all] [proto PROTO] [from ADDR [port PORT]] [to ADDR [port PORT]]
ufw deny from 12.34.56.78 #拒绝来自该IP的所有流量
ufw allow tcp from 12.34.56.78 to any port 80 #ip任意是写any，端口任意是不写
ufw allow in eth0 from 12.34.56.78 #加上网卡的限制

## 删除规则
## ufw delete NUM
ufw delete 2 #删除第二条规则

## 重载
## 假如修改了/etc/ufw中的rule文件，需要重载以使其生效
ufw reload

## 重置
ufw reset #删除所有设置的规则







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