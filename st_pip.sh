#安装
pip install x
#卸载
pip uninstall x

#手动设置镜像源
pip install x -i https://pypi.tuna.tsinghua.edu.cn/simple 
#全局设置源
'''
Linux:
~/.pip/pip.conf

Windows:
C:\Users\lychahaha\AppData\Roaming\pip\pip.ini
C:\Users\lychahaha\pip\pip.ini (新版)

[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
'''

#装特定版本
pip install x==1.2

#升级版本
pip install --upgrade x

#显示信息
pip list #列出已安装的包(包含版本信息)
pip list -o #列出可以升级的包
pip show x #显示包的信息


#离线安装
pip freeze > requirements.txt #将目前安装的库列出来
pip download -d save_dir -r requirements #根据requirements下载轮子（它会顺便下载这些库依赖的库，所以requirements可以只写你需要的核心库）
pip install --no-index --find-links=save_dir -r requirements #在离线电脑上安装
