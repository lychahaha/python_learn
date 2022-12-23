# systemctl
systemctl start xxx #启动服务
systemctl stop xxx #关闭服务
systemctl restart xxx #重启服务
systemctl enable xxx #开机启动
systemctl disable xxx #取消开机启动
systemctl status xxx #查看状态
systemctl daemon-reload #重新加载配置文件

# 配置文件
## 假设服务名为frpc，则需要在/etc/systemd/system中创建frpc.service文件
## frpc.service文件如下格式：
[unit]
Description=frpc Service
After=network.target
 
[Service]
Type=simple
Restart=on-failure
RestartSec=5s
ExecStart=/usr/local/bin/frpc -c /etc/frpc/frpc.ini
 
[Install]
WantedBy=multi-user.target
