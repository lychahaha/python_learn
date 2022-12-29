# 服务端

## 安装
apt install nfs-kernel-server

## 创建文件夹
mkdir /home/nfs
chmod -R 777 /home/nfs

## 修改nfs配置/etc/exports
/home/nfs *(rw,async,root_squash,no_subtree_check)
### *过滤地址，比如设置成192.168.24.*
### rw是读写，ro是只读
### async是异步写入硬盘，sync是同步写入硬盘
### root_squash是客户端root当成匿名用户，no_root_squash是当成root，all_squash是所有用户都是匿名用户（其余两个碰到服务端有同名的会当成相应的用户）
### no_subtree_check是不检查父目录的权限
### 这里写多行可以挂载多个文件系统

## 开机启动
systemctl enable rpcbind
systemctl enable nfs-server

## 固定mountd和nlockmgr的端口
### /etc/default/nfs-kernel-server
RPCMOUNTDOPTS = "--manage-gids -p 40001"
### /etc/modprobe.d/options.conf
options lockd nlm_udpport=40002 nlm_tcpport=40002
### /etc/modules
lockd

## 设置防火墙
firewall-cmd --zone=public --permanent --add-server=nfs
firewall-cmd --zone=public --permanent --add-server=rpc-bind
firewall-cmd --zone=public --permanent --add-port=40001/tcp

## 重启
reboot

## 查看端口是否固定
rpcinfo -p

## 查看nfs运行情况
nfsstat



# nfs客户端

## 安装
apt install nfs-common

## 查看nfs服务端情况
showmount -e 192.168.25.25

## 创建文件夹
mkdir /home/share

## mount挂载
mount -t nfs 192.168.3.3:/home/nfs /home/share

## fstab挂载:/etc/fstab
192.168.25.25:/home/nfs  /home/share  nfs defaults,_netdev  0  0




# ftp,[vsftpd],tftp

# [samclient],pdbedit