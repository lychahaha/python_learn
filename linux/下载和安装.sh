# yum
yum install xxx #下载并安装
yum install xxx.rpm #安装本地rpm包
    -y #自动确定
    --nogpgcheck #忽视安全性校验
yum remove xxx #卸载
yum update xxx #更新
yum info xxx #详细显示xxx的信息

yum search ping #根据需要的命令寻找需要安装的包

yum provides xx.repo #显示文件属于哪个包

yum clean all #清除所有元数据缓存
yum makecache #生成新元数据缓存
    # 配置源:/etc/yum.conf
    #        /etc/yum.repos.d/*.repo

yum repolist #列出在用源
yum repolist all #列出所有源
yum repolist disabled #列出禁用源

yum list all #获取所有包的信息
yum list all mysql* #使用通配符获取包信息
yum list installed #获取已安装的包
yum list available #获取未安装但可安装的包
yum list updates #获取可更新的包

##本地源配置文件示例
[centos7u4]  ##yum源区别名称，用来区分其他的yum源
name=centos7u4  ##yum源描述   yum源名字
baseurl=file:///mnt/centos7u4  ##指定本地yum源的路径
enabled=1  ##是否使用此yum源（1为打开，0为关闭）
gpgcheck=0 ##检查软件  （1是检查，0是不检查）
##网络源配置文件示例
[base]
name=CentOS7
baseurl="https://repo.huaweicloud.com/centos/\$releasever/os/\$basearch/"
enabled=1
gpgcheck=0

##离线安装
yum install yum-utils #先安装所需工具
repotrack xxx #下载全量依赖包（下载到当前目录）
yumdownloader --resolve xxx #下载当前机器缺失的依赖包（去掉--resolve则只下载目标包）
rpm -Uvh --force --nodeps *.rpm #离线机器上执行

# apt
apt install xxx #下载并安装
    -y #自动确定
apt remove xxx #卸载
    --purge #配置文件也删除
apt list --installed #列出已安装的包
apt update #更新源
    # 配置源:/etc/apt/sources.list

# wget
wget http://xxx.xxx.xxx/abc.tar.gz #下载互联网资源
    -c #断点续传
    -b #后台下载

# curl

# rpm
rpm -i xxx.rpm #安装rpm包
    -v -vv -vvv #显示不同程度的详细信息
    -h #显示安装进度
    --nodeps #忽略依赖关系
    --force #不管有没有装过都重新安装
rpm -e xxx #卸载
rpm -U xxx #升级
rpm -q xxx #查询xxx是否已安装
    -qi #查询xxx的详细信息
    -ql #查询xxx所包含的文件
    -qf #查询某个文件是哪个安装包生成的
    -qa #查询当前系统所有已安装的包

# 开发版
## 开发版指安装的包带源代码和库，可以方便后面用来编译
## yum对应的开发版是xxx-devel
## apt对应的开发版是xxx-dev
