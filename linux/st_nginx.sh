# 说明文档：nginx.org/en/docs
# 静态资源服务器、、反向代理、负载均衡

nginx -V #查看版本和模块信息
nginx -t #查看配置是否有问题
nginx -s reload #重新加载配置（重启worker）

nginx -s stop #停止运行
nginx -s quit #？

# conf
/etc/nginx/nginx.conf #全部配置
/etc/nginx/conf.d/xxx.conf #主机配置（每一个里面只需写server即可）

####/etc/nginx/conf.d/xxx.conf
#conf结构
## 全局设置
## 性能设置
## http设置
##  upstream
##  server（虚拟主机）
##   location （匹配规则）

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


# 可用信息参数
$remote_addr #客户端ip
$http_x_forwarded_for #客户端真实ip（有反向代理时）
$uri #域名后的资源路径
$scheme #请求的协议
$remote_user #客户端用户名
$time_local #访问时间和时区
$request #请求起始行信息（请求方法、http协议等）
$status #http状态码
$http_referer #客户端从哪个链接访问过来的（防盗链相关）
$http_user_agent #客户端用户信息
