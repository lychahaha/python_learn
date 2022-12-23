# configure
## 它是一个脚本
## 用于检测当前系统环境是否满足编译要求
## 生成Makefile
configure
configure --prefix=/usr #指定安装路径，写进Makefile，默认是/usr/local

# make
## 执行Makefile
make #编译
make install #将相关文件放进系统文件夹(bin、man、lib等)

# autoscan > autoscan.log configure.scan
# configure.scan -> configure.in
# aclocal > aclocal.m4 autom4te.cache/
# autoconf > configure
# autoheader > config.h.in
# Makefile.am
# automake > Makefile.in depcomp missing install-sh
# configure > Makefile
# make > 可执行文件

