#jh:func-base-mixsimilar,lazy-type
import importlib
import os,sys


'''外部导入'''
import xxx
import xxx as xx
import xxx.xx
from xxx import xx,yy
from xxx import xx as x
from xxx import *


'''内部导入'''

#import当前目录的module
from . import module
#import当前目录的package(就是下一级目录的__init__.py,只是它可以索引下级目录的其他module)
from . import package

#import当前目录的module里的变量
from .module import val
#import当前目录的package里的变量
from .package import val

#import下一级目录的module
from .package import module
#import下两级目录的module
from .p1.p2 import module

#import下一级目录的module里的变量
from .package.module import val

#import上一级目录的module
from .. import module

#import上两级目录的package里的变量(上一级目录的__init__.py里的变量)
from .. import val


'''变量导入(外部导入)'''
module = __import__('module')
package = __import__('package')
module = __import__('package.module', fromlist=True)


'''其他'''
#重载
importlib.reload(xxx)

#增加搜索路径
sys.path.append(path)
