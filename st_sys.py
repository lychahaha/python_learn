#coding=utf-8

import sys

#未捕获的异常的顶层钩子
#(exctype,value,traceback)->none
sys.excepthook
#默认钩子
sys.__excepthook__

#交互式的打印钩子
#obj->none
sys.displayhook
#默认钩子
sys.__displayhook__

#获取异常信息函数
#处理异常后仍然存在
#none->(exctype,value,traceback)
sys.exc_info()

#清理异常信息函数
sys.exc_clear()

#退出
#实际上会抛出异常
#status->none
sys.exit(0)

#python.exe文件路径
sys.executable

#python.exe所在目录
sys.exec_prefix#平台依赖?
sys.prefix#平台独立?

#递归最大深度
sys.getrecursionlimit()

#一个list,类库缓存
#import时最先查找
sys.modules

#一个list,存放finder对象
#import时modules找不到则在这里查找
#finder对象需实现find_module方法,如果找到则返回finder对象,这个对象有load_module方法
sys.meta_path

#一个list,存放python环境变量路径
#import时meta_path找不到则在这里查找
#在这里也找不到会抛出异常ImportError
sys.path

#交互式的提示符
sys.ps1#>>>
sys.ps2#...

