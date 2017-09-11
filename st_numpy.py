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
x = a.all()#交
x = a.any()#并

#求下标
b = a.argmax()
b = a.argmin()
b = a.argsort()#返回排好序的下标

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

#
c = np.maximum(a,b)
c = np.minimum(a,b)

#
c = np.clip(a, min_, max_)

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




#tmp
b = np.abs(a)
b = np.absolute(a)#支持复数
#np.absolute_import
c = np.add(a,b)
#np.add_docstring
#np.add_newdoc
#np.add_newdoc_ufunc
#np.add_newdocs
np.alen(a)#相当于shape[0]
np.all(a)
np.allclose(a,b)#比较两个数组的元素是否某种程度上接近
np.alltrue(a)
np.amax(a)#若有np.NaN,则返回np.NaN
np.amin(a)
np.angle(a, deg=False)#返回复数的角度,deg为true返回角度制,false返回弧度制
np.any(a)
np.append(a, b, axis=None)#按axis进行append,两数组维数应该一样,不给axis则先flat成一维
np.apply_along_axis(func, axis, a)#沿着某维度apply
np.apply_over_axes(func, a, axes)#沿着多个维度apply
np.arange(start, stop, step)
np.arccos(a)
np.arccosh(a)
np.arcsin(a)
np.arcsinh(a)
np.arctan(a)
np.arctan2(ay, ax)
np.arctanh(a)
np.argmax(a)
np.argmin(a)
#np.argpartition
np.argsort(a)
np.argwhere(a)#返回true的全下标
np.around(a, decimals=0)#四舍五入,decimals决定对几位小数进行四舍五入,decimals可以为负
np.array(l)
#np.array2string
np.array_equal(a, b)#==
np.array_equiv(a, b)#广播后==
#np.array_repr
np.array_split(a, n, axis=0)#split,n可以不整除维度的长度
#np.array_str
#np.asanyarray
np.asarray(a)
#np.asarray_chkfinite
#np.ascontiguousarray
#np.asfarray
#np.asfortranarray
np.asmatrix(a)
np.asscalar(a)
#np.atleast_1d
#np.atleast_2d
#np.atleast_3d
np.average(a, axis=None, weights=None)
#np.bartlett
#np.base_repr
#np.bench
#np.binary_repr
np.bincount(a, weights=None)#计算元素出现次数,a必须是1维非负整数数组
np.bitwise_and(a, b)#按位与,输入必须是int或bool
np.bitwise_not(a)
np.bitwise_or(a, b)
np.bitwise_xor(a, b)
#np.blackman
np.block(a)#类似分块矩阵的合并
#np.bmat
np.bool
#np.bool8
#np.bool_
#np.broadcast
#np.broadcast_arrays
np.broadcast_to(a, shape)
np.busday_count(date_beg, date_end, )#计算日期相差天数
#np.busday_offset
#np.busdaycalendar
np.byte#np.int8
np.byte_bounds(a)#返回(beg,end),该数组占用内存的起终指针
np.bytes0#np.btypes_
np.bytes_
#np.c_
np.can_cast(t1, t2)#判断是否能类型转换
#np.cast
np.cbrt(a)#立方根
np.cdouble#np.complex128
np.ceil(a)
np.cfloat#np.complex128
##np.char
#np.character
#np.chararray
np.choose(a, c)#[c[a[i]][i] for i in a.size]
np.clip(a, min_, max_)
np.clongdouble#np.complex128
np.clongfloat#np.complex128
np.column_stack(arrays)#np.concatenate(arrays,axis=1)
np.common_type(*arrays)#返回运算后会变成的类型
#np.compare_chararrays
np.compat
np.complex
np.complex128
np.complex64
np.complex_#np.complex128
#np.complexfloating
np.compress(conditions, a, axis=0)#根据条件对axis过滤
np.concatenate(arrays, axis=0)
np.conj(a)#np.conjugate
np.conjugate(a)#共轭
np.convolve(a, b)#1维卷积
np.copy(a)
np.copysign(a, b)#结合a的绝对值,b的正负号
#np.copyto
#np.core
#np.corrcoef#相似度
#np.correlate
np.cos(a)
np.cosh(a)
np.count_nonzero(a)
#np.cov
#np.cross
np.csingle#np.complex64
##np.ctypeslib
np.cumprod(a)#np.cumproduct
np.cumproduct(a)#前缀积
np.cumsum(a)
np.datetime64
#np.datetime_as_string
#np.datetime_data
np.deg2rad(a)
np.degrees(a)#rad2deg
np.delete(a, ixs, axis=None)#删除axis下某些下标对应的行,没axis时会flat
#np.deprecate
#np.deprecate_with_doc
np.diag(a)#输入什么输出另一个(对角线向量<->对角矩阵)
np.diag_indices(n, ndim=2)#生成对角线的分布下标
np.diag_indices_from(a)#生成该矩阵对角线的分布下标
np.diagflat(obj, k=0)#obj进行flat后,生成对角线矩阵,k是对角线偏移量,k=0表示obj是对角线
np.diagonal(a)#返回a的对角线,还有更复杂的功能
np.diff(a, n=1)#n阶差分
np.digitize(x, y, right=False)#x的每个元素在有序数组y中二分,返回在y中的下标,right为true表示下界,false表示上界
#np.disp
np.divide(a, b)
np.division(a, b)#整数除法
np.divmod(a, b)#返回商和余数
np.dot(a, b)
np.double#np.float64
np.dsplit(a, n)#np.split(a, n, axis=2)
np.dstack(arrays)#np.concatenate(arrays, axis=2)
np.dtype
np.e
np.ediff1d(a, to_end=None, to_begin=None)#a进行flat后求差分,并在首尾插入to_begin和to_end,它们可以是标量或数组
#np.einsum
#np.einsum_path
#np.emath
np.empty(shape)
np.empty_like(a)
np.equal(a, b)#==
#np.errstate
np.euler_gamma
np.exp(a)
np.exp2(a)
np.expand_dims(a, axis)#插入一个长度为1的维度
np.expm1(a)#np.exp(a)-1
np.extract(condition, a)#condition的shape和a的要相同,符合条件的会选出来变成1维数组,相当于a[b]
np.eye(n, M=None, k=0)#单位矩阵,n是行数,M是列数默认等于n,k是对角线偏移量
np.fabs(a)#不支持复数
#np.fastCopyAndTranspose
##np.fft(a)
np.fill_diagonal(a, x)#x填入a的对角线
#np.find_common_type
np.finfo(dtype)#浮点数的上下限等信息
np.fix(a)#往0的方向取整
#np.flatiter
np.flatnonzero(a)#返回flat后的a中非0的元素下标
#np.flexible
np.flip(a, axis)#reverse
np.fliplr(a)#np.flip(a, 1)
np.flipud(a)#np.flip(a, 0)
np.float
np.float16
np.float32
np.float64
np.float_#np.float64
np.float_power(a, b)#a**b
#np.floating
#np.floor
np.floor_divide(a, b)#np.floor(a/b)
np.fmax(a, b)#优先不返回np.nan
np.fmin(a, b)
np.fmod(a, b)#a%b
#np.format_parser
np.frexp(a)#返回(m,e),a=m*2**e,其中0<=m<1
np.frombuffer(buf, dtype=np.float, count=-1, offset=0)
np.fromfile(file, dtype=np.float, count=-1, sep='')
np.fromfunction(func, shape)#func的参数是下标,返回值
np.fromiter(iter, dtype, count=-1)
#np.frompyfunc
#np.fromregex
np.fromstring(str, dtype=np.float, count=-1, sep='')
np.full(shape, x)
np.full_like(a, x)
#np.fv
#np.generic
#np.genfromtxt
np.geomspace(start, stop, num, endpoint=True)#生成几何级数,endpoint表明是否右闭
#np.get_array_wrap
#np.get_include
#np.get_printoptions
#np.getbufsize
#np.geterr
#np.geterrcall
#np.geterrobj
np.gradient(a)#求梯度
np.greater(a, b)#>
np.greater_equal(a ,b)#>=
np.half#np.float16
#np.hamming
#np.hanning
np.heaviside(a, b)#a<0 -> 0, a==0 -> b, a>0 -> 1
#np.histogram
#np.histogram2d
#np.histogramdd
np.hsplit(a, n)#np.split(a, n, axis=1)
np.hstack(arrays)#np.stack(arrays, axis=1)
np.hypot(a, b)#sqrt(a**2+b**2)
#np.i0
np.identity(n)#单位矩阵
np.iinfo(dtype)#整数的信息
np.imag(a)#虚部
np.in1d(a, b, invert=False)#判断a中元素是否在b中,invert表示结果是否取反
#np.index_exp
np.indices(shape)#生成分布下标
#np.inexact
np.inf
np.info(obj)#输出帮助信息,obj是对象或者字符串
np.infty
np.inner(a, b)#内积
np.insert(a, ixs, val, axis=None)
np.int
np.int0#np.int64
np.int16
np.int32
np.int64
np.int8
np.int_#np.int32
#np.int_asbuffer
np.intc#np.int32
np.integer
#np.interp
#np.intersect1d
np.intp#np.int64
np.invert(a)#np.bitwise_not
#np.ipmt
#np.irr
#np.is_busday
np.isclose(a, b)#是否接近
np.iscomplex(a)
np.iscomplexobj(a)#至少有一个复数
np.isfinite(a)
np.isfortran(a)
np.isin(a, b, invert=False)#判断a中元素是否在b中,invert表示结果是否取反
np.isinf(a)
np.isnan(a)
np.isnat(a)#is not a time
np.isneginf(a)
np.isposinf(a)
np.isreal(a)
np.isrealobj(a)#是否完全没有复数
np.isscalar(a)
np.issctype(a)#is scalar type
#np.issubclass_
#np.issubdtype
#np.issubsctype
np.iterable(a)#是否能遍历
#np.ix_
#np.kaiser
np.kron(a, b)#kron积(张量积)
np.ldexp(a, b)#a*2**b
np.left_shift(a, b)#a<<b
np.less(a, b)#a<b
np.less_equal(a, b)#a<=b
np.lexsort(arrays)#按keys进行sort然后返回下标
##np.lib
##np.linalg
np.linspace(start, stop, num=50, endpoint=True, retstep=False)#生成算术级数(等差数列),retstep为true返回(数列,公差)
np.little_endian#true,小端格式
np.load(file, encoding='ACSII')
np.loads(file, encoding='ACSII')
#np.loadtxt(file, dtype=np.float)
np.log(a)#ln(x)
np.log10(a)#lg(x)
np.log1p(a)#ln(1+x)
np.log2(a)#log2(x)
np.logaddexp(a, b)#ln(e^a+a^b)
np.logaddexp2(a, b)#log2(2^a+2^b)
np.logical_and(a, b)
np.logical_not(a)
np.logical_or(a, b)
np.logical_xor(a, b)
np.logspace(start, stop, num=50, endpoint=True, base=10.0)#对算术级数取指数
np.long#np.int
np.longcomplex#np.complex128
np.longdouble#np.float64
np.longfloat#np.float64
np.longlong#np.int64
#np.lookfor
##np.ma
#np.mafromtxt
#np.mask_indices
np.mat(obj)
##np.math
np.matmul(a, b)
np.matrix(obj)
##np.matrixlib
np.max(a)#np.amax
np.maximum(a, b)
np.maximum_sctype(obj)#返回这个数据可能的最高类型
np.may_share_memory(a, b)#是否占用共同的内存
np.mean(a)#均值
np.median(a)#中位数
#np.memmap
#np.meshgrid
#np.mgrid
np.min(a)#amin
np.min_scalar_type(obj)#返回这个数据可能的最低类型
np.minimum(a, b)
#np.mintypecode
#np.mirr
np.mod(a, b)#a%b
np.modf(a)#返回(小数部分,整数部分)
np.moveaxis(a, src, dst)#高级转置
np.msort(a)#np.sort(a, axis=0)
np.multiply(a, b)
np.nan
np.nan_to_num(a)#把非数字改成极端数字
np.nanargmax(a)#不考虑np.nan
np.nanargmin(a)
np.nancumprod(a)#把np.nan当成1
np.nancumsum(a)#把np.nan当成0
np.nanmax(a)
np.nanmean(a)#不统计np.nan
np.nanmedian(a)
np.nanmin(a)
np.nanpercentile(a)
np.nanprod(a)
np.nanstd(a)
np.nansum(a)
np.nanvar(a)
np.nbytes#dict,记录类型所占字节数
np.ndarray
np.ndenumerate(a)#迭代器,生成(全下标,值)
#np.ndfromtxt
np.ndim(a)#维度数,a.shape.size
np.ndindex(*shape)#迭代器,生成全下标
#np.nditer
np.negative(a)#-a
#np.nested_iters
#np.newaxis
np.nextafter(a, b)#返回a向b方向的下一个浮点值
np.nonzero(a)#返回非0的分布下标
np.not_equal(a, b)#a!=b
#np.nper
#np.npv
#np.numarray
#np.number
np.obj2sctype(obj)#obj在np里的type
np.object
np.object0#np.object_
np.object_
#np.ogrid
#np.oldnumeric
np.ones(shape)
np.ones_like(a)
np.outer(a, b)#外积
#np.packbits#按位拼
#np.pad
#np.partition
np.percentile(a, q)#求百分数
np.pi
np.piecewise
np.pkgload
np.place
np.pmt
np.poly
np.poly1d
np.polyadd
np.polyder
np.polydiv
np.polyfit
np.polyint
np.polymul
np.polynomial
np.polysub
np.polyval
np.positive
np.power
np.ppmt
np.print_function
np.prod
np.product
np.promote_types
np.ptp
np.put
np.putmask
np.pv
np.r_
np.rad2deg
np.radians
np.random
np.rank
np.rate
np.ravel
np.ravel_multi_index
np.real
np.real_if_close
np.rec
np.recarray
np.recfromcsv
np.recfromtxt
np.reciprocal
np.record
np.remainder
np.repeat
np.require
np.reshape
np.resize
np.result_type
np.right_shift
np.rint
np.roll
np.rollaxis
np.roots
np.rot90
np.round
np.round_
np.row_stack
np.s_
np.safe_eval
np.save
np.savetxt
np.savez
np.savez_compressed
np.sctype2char
np.sctypeDict
np.sctypeNA
np.sctypes
np.searchsorted
np.select
np.set_numeric_ops
np.set_printoptions
np.set_string_function
np.setbufsize
np.setdiff1d
np.seterr
np.seterrcall
np.seterrobj
np.setxor1d
np.shape
np.shares_memory
np.short
np.show_config
np.sign
np.signbit
np.signedinteger
np.sin
np.sinc
np.single
np.singlecomplex
np.sinh
np.size
np.sometrue
np.sort
np.sort_complex
np.source
np.spacing
np.split
np.sqrt
np.square
np.squeeze
np.stack
np.std
np.str
np.str0
np.str_
np.string_
np.subtract
np.sum
np.swapaxes
np.sys
np.take
np.tan
np.tanh
np.tensordot
np.test
np.testing
np.tile
np.timedelta64
np.trace
np.tracemalloc_domain
np.transpose
np.trapz
np.tri
np.tril
np.tril_indices
np.tril_indices_from
np.trim_zeros
np.triu
np.triu_indices
np.triu_indices_from
np.true_divide
np.trunc
np.typeDict
np.typeNA
np.typecodes
np.typename
np.ubyte
np.ufunc
np.uint
np.uint0
np.uint16
np.uint32
np.uint64
np.uint8
np.uintc
np.uintp
np.ulonglong
np.unicode
np.unicode_
np.union1d
np.unique
np.unpackbits
np.unravel_index
np.unsignedinteger
np.unwrap
np.ushort
np.vander
np.var
np.vdot
np.vectorize
np.version
np.void
np.void0
np.vsplit
np.vstack
np.warnings
np.where
np.who
np.zeros
np.zeros_like