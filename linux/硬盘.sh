#ignore:fsck

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

# fdisk
fdisk -l #显示所有硬盘和分区信息(需要root权限)
fdksi /dev/sda #管理
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

# umount
umount /dev/sda1


# 硬盘、分区、路径的概念
## 硬盘 disk   /dev/sda
## 分区 device /dev/sda1
## 路径        /mnt/hehe
##
## 1.使用fdisk可以从硬盘创建多个分区
## 2.使用mkfs格式化分区
## 3.使用mount将分区挂载到具体某个路径
