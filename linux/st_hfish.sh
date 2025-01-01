# 安装
docker run -itd --name hfish \
--network host \
--privileged=true \
threatbook/hfish-server:latest

# 后台服务
# https://192.168.1.2:4433/web/
# 初始账号密码：admin HFish2021