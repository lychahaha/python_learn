cython a.py
gcc -c a.c -fPIC -I ~/anaconda3/include/python3.5m
gcc -shared a.o -o a.so

#NOTE
#1.名字不能改,原本是a.py,最后就要是a.so
#2.记得加个__init__.py在同级目录
