#jh:proc-base-typical,key-type
import cProfile
#import profile
import pstats

def main():
    xxx

#typical
if __name__ == '__main__':
    cProfile.run("main()", 'result.txt')

    p = pstats.Stats('result.txt')
    p.sort_stats('cumulative').print_stats(10)
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
    p = p.strip_dirs()#去掉无关的路径信息

    p = p.sort_stats(-1)#按module/name/line排序
    p = p.sort_stats('name')#按函数名字排序
    p = p.sort_stats('cumulative')#按累积运行时间倒序排序

    #输出
    p.print_stats()#打印所有信息
    p.print_stats(3)#打印3条
    p.print_stats(0.3)#打印30%

    p.print_callers("xxx")#打印调用了这个函数的函数
    p.print_callees("xxx")#打印这个函数调用了的函数



#test.py
@profile
def main():
    xxx
main()

#typical
kernprof -l -v test.py

kernprof -l -o xxx.lprof test.py
python -m line_profile xxx.lprof

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