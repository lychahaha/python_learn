#jh:func-base-mixsimilar,key-type

#coding=utf-8
import random

#[0,1)的浮点数
print random.random()

#[beg,end)的浮点数
print random.uniform(10,20)

#[beg,end]的整数
print random.randint(10,20)

#[beg,end,step)的整数
print random.randrange(10,20,2)

#序列中的一个元素
print random.choice([1,2,3,4,5])

#打乱一个序列
a = [1,2,3,4,5]
random.shuffle(a)
print a
