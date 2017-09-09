#coding=utf-8
from numpy import *

#基础类array数组
a = array([1,2,3])
a = array([[1,2],[3,4]], dtype=complex)


#属性
a.ndim#维度
a.shape#每个维度的长度(可写,元组形式)
a.dtype#元素类型
a.size#元素总个数,等于shape中元素之积
a.itemsize#元素占字节数


#改变形态的函数
b = a.reshape(x, y, z, ...)
a.resize( (x, y, z, ...) )#原地reshape
#转置
b = a.transpose()
#旋转
b = rot90(a, 3)#转3次
#翻转
b = fliplr(a)#左右翻转(次高维翻转)
b = flipud(a)#上下翻转(最高维翻转)
#滚动
b = roll(a, 3, axis=2)#沿着第2维滚动3格
b = roll(a, 3)#沿着展平迭代器滚动3格
#展平
b = a.ravel()
#组合
c = vstack((a,b))#(最高维度长度相加)
c = hstack((a,b))#(次高维度长度相加)
c = concatenate((a,b), axis=0)
#分割
b = hsplit(a, 3)#(次高维度被分割)
b = vsplit(a, 3)#(最高维度被分割)
#深拷贝
b = a.copy()


#聚类函数
x = a.sum()
x = a.max()
x = a.min()
x = a.mean()#均值
x = a.var()#方差
x = a.std()#标准差

#求下标
b = a.argmax()
b = a.argmin()

#axis参数能按维度聚类(其他同理)
b = a.sum(axis=0)

#前缀和
b = a.cumsum(axis=0)#按轴求前缀和
b = a.cumsum()#压成一维后求前缀和




#数值计算
b = a+1
b = a+a
#(加减乘除幂同理)

b = exp(a)
b = sin(a)
b = cos(a)
b = sqrt(a)

#逻辑运算
b = a<1
b = a<a
#(大于,等于,与或非...同理)


#矩阵运算
c = dot(a, b)#矩阵乘法
x = trace(a)#矩阵的迹
b = eye(2)#单位矩阵
b = diag(a)#输入什么输出另一个(对角线向量<->对角矩阵)


#分片索引
b = a[2:9:2]
b = a[0:5,1:3,4]
b = a[1,2,...]#除后省略,还可以前省略,中间省略
c = b[a]


#生成函数(一般默认float64)
b = zeros((3,4))#全零
b = ones((2,3,4), dtype=int32)#全1
b = empty((2,3))#随机初值
b = full((2,3), 5)#全5

b = zeros_like(a)#相当于zeros(a.shape)
b = ones_like(a)
b = empty_like(a)
b = full_like(a, 5)

b = identity(3)#单位矩阵

b = repeat(a, 4)#相当于遍历a的元素,并填充4次

b = arange(beg,end,step)#等差数列(固定差)(默认int32)
b = linspace(beg,end,num)#等差数列(固定项数)

b = fromfunction(fx, shape)#fx输入坐标,输出数值


#迭代器
a.flat#迭代每个元素


#数据类型
float16,float32,float64(float)
int8,int16,int32(int),int64
uint8,uint16,uint32(uint),uint64
complex64,complex128(complex)
bool
str

#其他
b = a.astype(np.uint8)#类型转换


#线性代数
from numpy import linalg

b = linalg.inv(a)#求逆
b = linalg.solve(a, y)#解方程
vals,vectors = linalg.eig(a)#特征值和特征向量矩阵
x = linalg.det(a)#行列式
x = linalg.matrix_rank(a)#秩

q,r = linalg.qr(a)#QR分解
u,s,v = linalg.svd(a)#SVD分解


#随机
from numpy import random

#设置随机种子
random.seed(2333)

#01浮点数均匀分布
b = random.rand(2,3,4)#参数是*shape
b = random.random((2,3))#参数是shape
#浮点数均匀分布
b = random.uniform(2,5,size=(2,3))#值域[2,5)
#整数均匀分布
b = random.randint(2,5,size=(2,3))#值域[2,5)
#正态分布
b = random.normal(loc,scale,size)#loc是均值,scale是标准差
#二项分布
random.binomial(n=5, p=0.5, size)#n次,每次概率为p

#采样
b = random.choice(a, 7)#有放回,取7个
b = random.choice(a, 7, replace=False)#无放回

#乱序
b = random.shuffle(a)#只打乱a的最高维

#排列
b = random.permutation(10)#10项的随机全排列
b = random.permutation(a)#相当于shuffle