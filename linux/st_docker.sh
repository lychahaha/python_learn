#ignore:Jenkins,Consul,Swarm,docker API

# 安装
## ubuntu
apt install docker.io docker-compose
## centos
yum install docker-ce docker-ce-cli containerd.io


# 设置镜像源
vim /etc/docker/daemon.json
'''
{
  "registry-mirrors": ["https://ckdhnbk9.mirror.aliyuncs.com"]
}
'''
'''
网易:http://hub-mirror.c.163.com
中科大镜像地址:http://mirrors.ustc.edu.cn/
中科大github地址:https://github.com/ustclug/mirrorrequest
Azure中国镜像地址:http://mirror.azure.cn/
Azure中国github地址:https://github.com/Azure/container-service-for-azure-china
DockerHub镜像仓库: https://hub.docker.com/ 
阿里云镜像仓库: https://cr.console.aliyun.com 
google镜像仓库: https://console.cloud.google.com/gcr/images/google-containers/GLOBAL 
coreos镜像仓库: https://quay.io/repository/ 
RedHat镜像仓库: https://access.redhat.com/containers
'''
systemctl restart docker


# 运行
docker run -i -t ubuntu /bin/bash #交互式运行
docker run -d ubuntu nginx #后台运行
    -t #前台运行
    -i #设置标准输入
    -d #后台运行
    -p 8080:80 #映射端口（宿主机：虚拟机），还可以-p 127.0.0.1:8080:80的形式选定宿主机ip
    --name xxx #给容器命名
    --net nxx #给容器指定网络
    --link mysql:sql #链接别的容器(容器名：别名)
    --add-host xxx:127.0.0.1 #手动添加host别名



# 容器相关命令
## 变更状态
docker start xxx #启动容器
docker stop xxx #关闭容器
docker restart xxx #重启容器
docker rm xxx #删除容器
## 状态查询
docker top xxx #查看容器进程信息
docker stats xxx #查看容器进程资源占用情况
docker inspect xxx #查看容器信息
docker logs xxx #查看容器log
    -f #追尾更新模式
    -t #加上时间
docker port xxx #查看端口占用情况
## 执行命令
docker exec -d xxx touch /hehe.txt #后台执行
docker exec -t -i xxx /bin/bash #前台执行
## 回到前台
docker attach xxx
## 复制文件
docker cp a.txt xxx:/etc/a.txt #从宿主机到容器xxx
docker cp xxx:/etc/a.txt a.txt #从容器xxx到宿主机



# 镜像相关命令
## 搜索镜像
docker search ubuntu
## 拉取镜像
docker pull ubuntu:12.04
docker pull abc/ubuntu
## 构建镜像
docker build -t "myname/imgname:v1"
    --no-cache #不要使用缓存
docker commit 4aab3 myname/imgname
    -m "commit info"
## 查看镜像历史
docker history 4aab3
## 推送镜像
docker push myname/imgname
## 删除镜像
docker rmi myname/imgname




# 网络相关命令
## 创建网络
docker network create nxx 
## 查看网络信息
docker network inspect nxx 
## 某个容器连接到网络中
docker network connect nxx xxx 
## 某个容器从网络中断开
docker network disconnect nxx xxx 




# 全局相关命令
docker info #查看docker信息
docker images #查看所有镜像
docker ps #查看所有容器
    -a #包括未在运行的
docker network ls #查看所有网络



# Dockerfile
FROM ubuntu:14.04
MAINTAINER myname "xx@123.com"
RUN apt install -y nginx
EXPOSE 80
CMD ["/bin/bash", "-l"]
ENTRYPOINT ["/bin/bash"]
WORKDIR /opt/db
ENV SS_PATH /home/ss
USER hehe
VOLUME ["/opt/project"]
ADD hehe.txt /opt/h.txt
COPY hehe.txt /opt/h.txt
LABEL
STOPSIGNAL
ARG
ONBUILD





# docker-compose
docker-compose up -d #构建并运行
docker-compose ps
docker-compose logs
docker-compose stop
docker-compose rm
