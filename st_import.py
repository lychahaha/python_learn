#coding=utf-8

import xxx
from xxx import xxx,yyy
from xxx import *

#import子模块(目录)里的__init__.py
import son_package

#import子模块(目录)夹里的一般py文件
from son_package import son_py
import son_package.son_py

#从子模块(目录)的__init__.py文件里import
from son_package import son_val

#import当前模块(目录)里的一般py文件
import cur_py

#从当前模块(目录)的__init__.py文件里import
from . import cur_val

#从当前模块(目录)的xxx.py里import
from cur_py import cur_val
from .cur_py import curval#有__init__.py

#从上一级目录的xxx.py里import
from ..cur_py import cur_val

