#ignore:fsck,[autofs]

# df
df #查看所有文件系统使用情况(包括占用空间)
    -h #友好地显示占用空间的单位
    -a #列出包括隐藏的文件系统
    -T #显示文件系统类型

# du
du xxx #打印xxx目录下所有子目录的空间占用情况
    -s #只打印xxx目录的占用情况
    -h #友好地显示占用空间的单位
    -a #也打印文件的占用情况
du -sh #查看当前目录的空间占用

# dd
dd if=/dev/zero of=xxx count=1 bs=100M #创造指定大小的文件

# blkid
blkid /dev/sdb1 #查看设置的UUID

# fdisk
fdisk -l #显示所有硬盘和分区信息(需要root权限)
fdisk /dev/sda #管理
    m #help
    p #打印分区信息
    n #新增分区
    d #删除分区
    q #退出
    w #保存并退出

# mkfs
mkfs -t ext3 /dev/sda3 #格式化分区

# mount
mount /dev/sda1 /mnt/xxx #挂载分区
mount -t nfs 192.168.2.2:/home/nfs /home/share #挂载nfs
mount UUID=xxxx /mnt/xxx #使用UUID挂载
mount -a #手动挂载fstab里未挂载的文件系统

# umount
umount /dev/sda1

# mkswap
mkswap /dev/sdb2 #格式化成swap分区

# swapon
swapon /dev/sdb2 #启用swap分区


# xfs_quota,edquota

# [vdo],vdostats,udevadm settle

# [raid],mdadm

# [LVM],pv*,vg*,lv*


# 硬盘、分区、路径的概念
## 硬盘 disk   /dev/sda
## 分区 device /dev/sda1
## 路径        /mnt/hehe
##
## 1.使用fdisk可以从硬盘创建多个分区
## 2.使用mkfs格式化分区
## 3.使用mount将分区挂载到具体某个路径


# fstab常见用法
/dev/sdb1               /mnt/xxx    ext4    defaults            0   0
192.168.2.2:/home/nfs   /home/share nfs     defaults,_netdev    0   0
UUID=xxxxxx             /mnt/cdrom  iso9660 defaults            0   0
