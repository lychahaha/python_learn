#ignore:newusers,pwunconv,chpasswd,pwconv

# useradd
useradd xxx #新增用户
    -d /home/xxx #指定home目录
    -g xxx #指定所属组
    -s xxx #指定shell
    -m # 使用-d指定原本不存在的目录时，要用这个创建新目录

# userdel
userdel xxx #删除用户
    -r #连同home目录一起删除

# usermod
usermod xxx -x xx #修改用户选项
    -d xxx #修改home目录
    -g xxx #修改所属组
    -s xxx #修改shell

# passwd
passwd #修改密码
passwd xxx #修改某个用户密码(root才有该权限)
passwd xxx -l #锁定用户使其不能登录

# groupadd
groupadd xxx #新增组

# groupdel
groupdel xxx #删除组

# groupmod
groupmod xxx -x xx #修改组选项
    -n xxx #修改组名字

# su
su #登录成root
su xxx #登录成xxx

# sudo
sudo cmd #以root权限执行命令

# newgrp
newgrp xxx #切换当前用户所属组(前提是当前用户属于多个组，切换到其中一个)

# chmod
chmod code xxx #改变xxx的权限为code
chmod [ugoa] [+-=] [rwxXstugo] #改变权限
    u #用户
    g #组
    o #其他
    a #全部
    + #加上权限
    - #减去权限
    = #权限赋值
## code的格式如777,每个数字分别代表本用户、所属组、其他人的权限码
## 每个权限码=4(r)+2(w)+1(x)


# 相关文件
/etc/passwd #存放了用户相关选项(除了密码)
/etc/shadow #存放了用户密码
/etc/group #存放了组相关选项


