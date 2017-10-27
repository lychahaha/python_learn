#jh:proc-base-typical,key-type

#基于函数的性能分析

import cProfile
#import profile
import pstats

def main():
    xxx

#typical
if __name__ == '__main__':
    cProfile.run("main()", 'result.txt')

    p = pstats.Stats('result.txt')
    p.sort_stats('cumtime').print_stats(10)
    p.print_callees("main")
#full
if __name__ == '__main__':
    #run
    #main()
    cProfile.run("main()", 'result.txt')

    #analyse

    #读入
    p = pstats.Stats('result.txt')

    #操作
    p.strip_dirs()#去掉filename的路径

    #排序
    p.sort_stats(0)#按调用次数倒序排序
    p.sort_stats('cumtime')#按累积运行时间倒序排序
    p.sort_stats('ncalls', 'cumtime')#多个key排序
    p.reverse_order()#倒序
    '''
    [2]cumtime,cumulative:cumtime,倒序
    [0]calls,ncalls:ncalls,倒序
    [1]time,tottime:tottime,倒序
    line:行号,顺序
    pcalls:调用次数(后面的数字???),倒序
    file,filename,module:文件名,顺序
    name:函数名,顺序
    nfl:(name/file/line),顺序
    [-1]stdname:标准函数名,顺序
    '''

    #输出
    p.print_stats()#打印所有信息
    p.print_stats(3)#打印3条
    p.print_stats(0.3)#打印30%
    p.print_stats('run')#打印正则匹配的函数
    p.print_stats('__init__', 5)#正则匹配后,打印前5条

    p.print_callers("xxx")#打印调用了这个函数的函数
    p.print_callees("xxx")#打印这个函数调用了的函数


#ncalls:调用次数
#tottime:不包括内部函数运行的总运行时间
#percall:tottime/ncalls
#cumtime:包括内部函数运行的总运行时间
#percall:cumtime/ncalls








#基于行的性能分析

#typical

#test.py
@profile
def main():
    xxx
main()

kernprof -l -v test.py

kernprof -l -o xxx.lprof test.py
python -m line_profile xxx.lprof
#note:它的代码输出是通过读入当前的代码文件得到的,而不是运行时候记录的


#full
kernprof:
-l #使用line_profile(一般都要)
-v #程序结束后打印结果
-o xxx.lprof #结果保存在文件里

#line:行号
#hits:调用次数
#time:总时间
#per hit:平均时间
#%time:总时间百分比
#line contents:该行内容


#保证profile不会报错
if 'builtins' not in dir() or not hasattr(builtins, 'profile'):
    import builtins
    def profile(func):
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        return inner
    builtins.__dict__['profile'] = profile








#针对python部分builtin和numpy的加速

from numba import jit

#typical
@jit
def test():
    ...
