#安装
pip install x
#卸载
pip uninstall x

#手动设置镜像源
pip install x -i https://pypi.tuna.tsinghua.edu.cn/simple 
#全局设置源
'''
~/.pip/pip.conf

[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
'''

#装特定版本
pip install x==1.2

#升级版本
pip install --upgrade x

#显示信息
pip list #列出已安装的包(包含版本信息)
pip list -o #列出可以升级的包
pip show x #显示包的信息

