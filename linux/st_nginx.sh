# 说明文档：nginx.org/en/docs
# 静态资源服务器、反向代理、负载均衡

nginx -V #查看版本和模块信息
nginx -t #查看配置是否有问题
nginx -s reload #重新加载配置（重启worker）

nginx -s stop #停止运行
nginx -s quit #？

# conf
/etc/nginx/nginx.conf #全部配置
/etc/nginx/conf.d/xxx.conf #主机配置（每一个里面只需写server即可）

#conf结构
## main上下文
## |-events上下文
## |-stream上下文（传输层）
##   |-server（具体服务，绑定端口）
## |-http上下文（应用层）
##   |-server（具体服务，绑定端口）
##     |-location（匹配规则）
##       |-if（判断）
##       |-limit_except（限制http方法）
##
## 不同server可以监听同一端口，但域名不同，nginx会判断同一端口哪个server更符合，若所有server都不符合，nginx会分配给该端口的默认server
##
## 上下文中包含单行指令和块指令，不同指令能出现的上下文是不同的(下文用“指令[上下文]”表示)，更深的上下文中的指令覆盖浅上下文的指令。
## 指令中可能会使用到变量，变量以$开头标识。

# events
events {}

# http
http {}

# stream
stream {}

# server
server {}

# location
location / {} #指定任意uri
location /abc/ {} #指定前缀
location ~ \.jpg #使用正则表达式（如果是~*，则是表示不区分大小写）
location = /abc {} #精确指定uri

# if
if ($t=1) {}

# limit_except[location]
limit_except GET {} #限制只能用GET


# listen[server]
listen 80; #监听80端口
listen 12.34.56.78:80; #指定ip地址
listen 80 default_server; #设置为该端口的默认server
listen 80 ssl; #开启ssl
listen 80 reuseport; #允许多个进程监听同一个端口，以提高并发率
listen 80 udp; #监听udp（仅stream可用）
# server_name[server]
server_name a.com; #监听域名
server_name a.com b.com; #监听多个域名
server_name *.a.com;
server_name ~^www\d+\.example\.com$; #使用正则表达式

# root[http,server,location,if]
root /var/www/nginx; #设置uri在服务器文件系统的root目录
# index[http,server,location]
index index.html; #指定默认文件
# proxy_pass[location,if,limit_except]
proxy_pass http://192.168.1.2:8080 #代理的核心指令(uri保持不变)
proxy_pass http://192.168.1.2:8080/aa/ #location匹配的uri部分会被/aa/替代
proxy_pass http://192.168.1.2:8080/ #特殊地，location匹配的部分会被去掉
proxy_pass https://192.168.1.2:8443 #以https代理
# return[server,location,if]（执行该命令后立即返回响应）
return 404; #报404错
return 200 "ok"; #加上文本内容
return https://xxx.com; #临时重定向（此时返回302）
return 301 https://xxx.com; #永久重定向
# rewrite[server,location,if]（在location中，每当重写完并执行完location后续指令，nginx会根据新uri重新匹配location，最多循环10次）
rewrite ^/old/(.*)$ /new-path/$1; #重写uri，将/old/xx/xx..变成/new/xx/xx..，重写后proxy_pass的uri部分将被忽略
rewrite ^/old/(.*)$ /new-path/$1 last; #立即结束该location，重新匹配location
rewrite ^/old/(.*)$ /new-path/$1 break; #立即结束该location，重新最后一次匹配location，不再允许rewrite
# break[server,location,if]（在location中，执行该指令后立即结束该location，重新最后一次匹配location，不再允许rewrite
break;
# alias[location]
# try_files[server,location]

# set[server,location,if]
set $my_name "haha"; #设置自定义的变量

# proxy_bind[http,server,location]
proxy_bind 192.168.2.1 #以另一个地址去代理
proxy_bind $remote_addr transparent #以客户端地址去代理，实现透明代理（需要防火墙和路由表配合）

# client_max_body_size[http,server,location]
client_max_body_size 10M; #客户端请求体最大大小，默认为1M，如果设为0则是允许无穷大

# error_page[http,server,location,if]
error_page 404 /404.html; #指定404的页面
error_page 401 402 403 /40x.html; #指定多个错误的页面
error_page 404 http://192.168.1.2:80; #错误重定向到链接
## 495 496 497表示客户端双向认证失败




# ssl_certificate[http,server]
ssl_certificate /etc/nginx/cert.crt; #证书
# ssl_certificate_key[http,server]
ssl_certificate_key /etc/nginx/pri.key; #私钥
# ssl_protocols[http,server]
ssl_protocols TLSv1.2 TLSv1.3; #指定tls版本
# ssl_verify_client[http,server]
ssl_verify_client on; #打开客户端双向认证
# ssl_client_certificate[http,server]
ssl_client_certificate /etc/nginx/cert.crt; #认证客户端证书的根证书

# auth_basic[http,server,location,limit_except]
auth_basic "Welcome" #开启登录认证并且设置认证问候语
# auth_basic_user_file[http,server,location,limit_except]
auth_basic_user_file conf.d/passwd.txt #设置账号密码表
## 该表每行是一个账号，格式为name:pass，注意这个pass是简单加密的字符串，使用openssl passwd my_password进行简单加密。

# allow[http,server,location,limit_except](充当防火墙过滤功能,按顺序自上而下检查)
allow 192.168.1.1; #允许该ip访问
allow 192.168.1.0/24; #允许该子网访问
allow all; #允许所有ip访问
# deny[stream,server]
deny 192.168.1.1; #与allow类似
deny all; #一般最下面加这个作为兜底

# log_format[http]
log_format my_log_format '$remote_addr [$time_local] $status $http_x_forwarded_for'; #声明日志格式，并设置该格式别名为my_log_format
# access_log[http,server,location,if,limit_except]
access_log /var/log/nginx/access.log my_log_format; #设置访问日志路径（my_log_format是log格式别名）
# error_log[main,http,mail,stream,server,location]
error_log /var/log/nginx/error.log info; #设置错误日志路径（错误等级包括debug,info,notice,warn,error,crit,alert,emerg） 

# user[main]
user root; #指定启动nginx的linux用户
# worker_processes[main]
worker_processes 2; #设置多少个worker进程
worker_processes auto; #根据cpu核数进行设置
# include[any]
inclue xx.conf;
# load_module[main]
load_module xx.so;

# charset

# upstream[http]
upstream xxx {}
# server[upstream]
server 192.168.1.2; #默认轮询
server 192.168.1.2:80; #加上端口
server 192.168.1.2 backup; #热备份，其他都坏了才用它
server 192.168.1.2 weight=2; #加权
# ip_hast[upstream]
ip_hash; #根据ip hash决定
# least_conn[upstream]
least_conn; #优先选连接数少的



# 常用参数
$remote_addr #ip包的源ip
$remote_port #传输层包的端口
$host #客户端请求的地址/域名
$http_host #客户端请求的地址/域名+端口
$http_x_forwarded_for #客户端真实ip（有反向代理时）
$request_uri #包含url参数的uri，始终是客户端的原始uri
$uri #不包含url参数的uri，被rewrite修改后该值会发生变化
$scheme #请求的应用层协议（http或https）
$protocol #请求的传输层协议（tcp或udp）
$remote_user #客户端用户名(如果没配置登录系统则为空)
$request #请求起始行信息（请求方法、http协议等）
$request_method #请求方法
$request_length #请求报文的长度
$http_referer #客户端从哪个链接访问过来的（防盗链相关）
$http_user_agent #请求头中的user-agent字段（和$user_agent是一个东西）
$status #响应状态码
$time_local #访问时间和时区







# 全局设置
user xxx; #使用什么用户运行nginx
worker_processes auto; #设置多少个worker进程

# http设置
http{
    log_format main '$remote_addr [$time_local] $status $http_x_forwarded_for'; #声明日志格式，并设置该格式别名为main

    client_max_body_size 10M; #这个值默认只有1M

    upstream vir_host {
        server 192.168.1.3:80;
        server 192.168.1.4:80;
    }

    server {
        # ...
    }
}

# stream设置(tcp/udp,与http并列)
stream{
    server {
        # ...
    }
}

# server
server {
    listen 8001; #监听的端口
    listen 443 ssl; #https必须加ssl
    listen 12.34.56.78:8001; #监听ip+端口（操作系统层面的匹配）
    server_name 12.34.56.78 abc.com; #匹配的ip/域名（可以写多个）（首尾可加*通配符）（支持正则表达式）

    charset utf-8; #使nginx支持中文路径

    access_log /var/log/nginx/xxx/xx.log main; #设置访问日志（main是log格式别名）
    error_log /var/log/nginx/xxx/xx.log info; #设置错误日志（错误等级包括debug,info,notice,warn,error,crit,alert,emerg） 
    # access_log off; #关闭访问日志

    error_page 404 /404.html; #设置错误页面的uri
    error_page 497 https://$host:1234$request_uri #端口为https时使用http请求的错误（可用于http重定向https）

    ssl_certificate /etc/nginx/ssl_certs/cert.crt; #ssl所需证书
    ssl_certificate_key /etc/nginx/ssl_certs/pri.key; #ssl所需密钥
    ssl_protocols TLSv1.2 TLSv1.3;
    
    # 静态资源服务器
    location / {
        root /home/nginx/data; #设置数据的root目录
        index index.html; #默认的index文件
    }

    # 反向代理服务器
    location / {
        proxy_pass http://192.168.1.3; #注意最后有没有/是很复杂的情况
        proxy_redirect http://192.168.1.3/ http://12.23.45.67:2333/; #重写响应头的location和refresh
        proxy_set_header Host $host; #?
    }

    # 负载均衡服务器
    location / {
        proxy_pass http://vir_host;
    }

    # 防盗链
    location ~* .*\.(gif|jpg|png)$ {
        root /home/nginx/img;
        valid_referers none blocked xx.com;
        if ($invalid_referer) {
            rewrite ^/ http://xx:9999/error.webp;
            break;
        }
    }
}

# https 代理
server{
    listen 1234 ssl;

    ssl_certificate /etc/nginx/ssl_certs/cert.crt; #ssl所需证书
    ssl_certificate_key /etc/nginx/ssl_certs/pri.key; #ssl所需密钥
    ssl_protocols TLSv1.2 TLSv1.3;

    # 开启双向认证
    ssl_client_certificate /etc/nginx/ssl_certs/ca.crt; #认证客户端所用的证书
    ssl_verify_client on;
    error_page 495 496 497 https://www.xxx.com; #认证失败后跳转到其他地方

    location / {
        proxy_pass http://127.0.0.1:4567;
    }
}


# tcp/udp server
server{
    listen 2345; #tcp
    proxy_pass 1.2.3.4:5678;
    
    listen 2345 udp; #udp
    
    proxy_connect_timeout 30s; #连接时的timeout(仅tcp有效)
    proxy_timeout 24h; #keep alive时间(udp默认10分钟断开)
    proxy_responses 1; #收到发回的1个udp包后断开(仅udp有效)
}



# https代理
server{
    listen 80;
    return 301 https://$host$request_uri;
}
server{
    listen 443 ssl;

    ssl_certificate /etc/nginx/cert.crt; #ssl所需证书
    ssl_certificate_key /etc/nginx/pri.key; #ssl所需密钥
    ssl_protocols TLSv1.2 TLSv1.3;

    root /var/www/html;
    index index.html;
}


# tcp/udp透明代理
## 原理：与普通反向代理类似，一方面与客户端建立连接，另一方面与服务端建立连接。
##      但是与服务端建立连接时伪装成客户端的IP进行发包，并且需要用防火墙和路由表截获服务端响应包。
## nginx.conf
user root; #最外面要指定用root
## stream.conf
server {
    listen 1234;
    proxy_pass 192.168.1.2:2345;
    proxy_bind $remote_addr transparent; #伪装成客户端IP发包给服务端
}
## 防火墙和路由设置
iptables -t mangle -R PREROUTING 1 -p tcp -s 192.168.1.2 --sport 2345 -j MARK --set-mark 1 #所有服务端服务端口发来的包都打标记1
ip route add local 0.0.0.0/0 dev lo table 100 #table100的都路由到本地
ip rule add fwmark 1 table 100 #标记为1的到table100处找路由规则


# http密码认证
## nginx.conf
server{
    auth_basic "Welcome" #开启登录认证并且设置认证问候语
    auth_basic_user_file conf.d/passwd.txt #设置账号密码表
}
## conf.d/passwd.txt
name1:pass1_encrypted
name2:pass2_encrypted


# 防火墙过滤
server{
    allow 192.168.1.0/24;
    allow 12.34.56.78;
    deny all;
}


# 负载均衡
## 轮询（默认）
upstream vir_host {
    server 192.168.1.3:80;
    server 192.168.1.4:80;
}
## 热备份
upstream vir_host {
    server 192.168.1.3:80;
    server 192.168.1.4:80 backup;
}
## 加权
upstream vir_host {
    server 192.168.1.3:80 weight=1;
    server 192.168.1.4:80 weight=2;
}
## 根据ip hash决定
upstream vir_host {
    server 192.168.1.3:80;
    server 192.168.1.4:80;
    ip_hash;
}
## 优先选连接数少的
upstream vir_host {
    server 192.168.1.3:80;
    server 192.168.1.4:80;
    least_conn;
}
## 优先选响应时间小的（第三方）
upstream vir_host {
    server 192.168.1.3:80;
    server 192.168.1.4:80;
    fair;
}


## return
## 一般用于各种类型码的重定向
server {
    listen 80;
    server_name hehe.com;
    return 301 https://www.hehe.com$request_uri; #80-http重定向443-https
}
## rewrite
## try_files
location {
    try_files $uri /index.php; #先尝试访问，不行就重定向到主页
}


