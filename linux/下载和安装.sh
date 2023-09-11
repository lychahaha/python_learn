# yum
yum install xxx #下载并安装
    -y #自动确定
yum remove xxx #卸载

# apt
apt install xxx #下载并安装
    -y #自动确定
apt remove xxx #卸载
apt list --installed #列出已安装的包

# wget
wget http://xxx.xxx.xxx/abc.tar.gz #下载互联网资源
    -c #断点续传
    -b #后台下载

# curl

# 开发版
## 开发版指安装的包带源代码和库，可以方便后面用来编译
## yum对应的开发版是xxx-devel
## apt对应的开发版是xxx-dev
