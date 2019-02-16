pip install x
pip install x -i https://pypi.tuna.tsinghua.edu.cn/simple #手动设置镜像源

#全局设置源
'''
~/.pip/pip.conf
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
'''

pip install x=1.23
