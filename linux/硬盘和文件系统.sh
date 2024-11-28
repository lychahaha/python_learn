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

# lsblk
lsblk #查看分区表信息（包括逻辑卷）



# 硬盘、分区、路径的概念
## 硬盘 disk   /dev/sda
## 分区 device /dev/sda1
## 路径        /mnt/hehe
##
## 1.使用fdisk可以从硬盘创建多个分区
## 2.使用mkfs格式化分区
## 3.使用mount将分区挂载到具体某个路径

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


# LVM
## 上述硬盘、分区、路径的操作是非LVM方式管理磁盘
## LVM方式是硬盘、分区(物理卷)、卷组、逻辑卷、路径
## 硬盘         disk        /dev/sda
## 分区(物理卷)  device(pv)  /dev/sda1
## 卷组         vg          ubuntu-root
## 逻辑卷       lv          /dev/mapper/ubuntu
## 路径                     /mnt/hehe
##
## 1.使用fdisk可以从硬盘创建多个分区
## 2.使用pvcreate将分区“格式化”为物理卷
## 3.使用vgcreate创建卷组，并将某个物理卷加入到卷组
## 4.使用lvcreate创建逻辑卷
## 5.使用mount将逻辑卷挂载到具体某个路径

# pv
pvcreate /dev/sda3 #创建物理卷
pvscan #查看物理卷简要信息
pvdisplay #查看物理卷详细信息

# vg
vgcreate vg_name /dev/sda3 #创建卷组（至少要将一个物理卷加入到卷组）
vgextend vg_name /dev/sda4 #添加物理卷到卷组
vgscan #查看卷组简要信息
vgdisplay #查看卷组详细信息

# lv
lvcreate -L 10G -n lv_name vg_name #将创建一个类似/dev/sda1分区文件，存储在/dev/mapper中
lvextend -L +5G /dev/mapper/vg_name_lv_name #逻辑卷扩容
lvscan #查看逻辑卷简要信息
lvdisplay #查看逻辑卷详细信息


# 其他
resize2fs /dev/mapper/vg_name_lv_name #刷新逻辑卷文件系统的大小
partprobe /dev/sda #告知内核该硬盘的分区表更新了


# xfs_quota,edquota

# [vdo],vdostats,udevadm settle

# [raid],mdadm


# fstab常见用法
/dev/sdb1               /mnt/xxx    ext4    defaults            0   0
192.168.2.2:/home/nfs   /home/share nfs     defaults,_netdev    0   0
UUID=xxxxxx             /mnt/cdrom  iso9660 defaults            0   0
