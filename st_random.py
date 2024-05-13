#jh:func-base-mixsimilar,key-type
import random

#[0,1)的浮点数
random.random()

#[beg,end)的浮点数
random.uniform(10,20)

#[beg,end]的整数
random.randint(10,20)

#[beg,end,step)的整数
random.randrange(10,20,2)

#序列中的一个元素
random.choice([1,2,3,4,5])

#打乱一个序列
a = [1,2,3,4,5]
random.shuffle(a)





import uuid

s = uuid.uuid4() #生成一个uuid