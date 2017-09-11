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
b = np.absolute(a)
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
np.diagflat(obj, k=0)#把obj进行flat后,生成对角线矩阵,k是对角线偏移量,k=0表示obj是对角线
np.diagonal(a)#返回a的对角线,还有更复杂的功能
np.diff(a, n=1)#n阶差分
np.digitize()
np.disp()
np.divide(a, b)
np.division(a, b)#整数除法
np.divmod(a, b)#返回商和余数
np.dot(a, b)
np.double#np.float64
np.dsplit
np.dstack
np.dtype
np.e
np.ediff1d
np.einsum
np.einsum_path
np.emath
np.empty
np.empty_like
np.equal
np.errstate
np.euler_gamma
np.exp
np.exp2
np.expand_dims
np.expm1
np.extract
np.eye
np.fabs
np.fastCopyAndTranspose
np.fft
np.fill_diagonal
np.find_common_type
np.finfo
np.fix
np.flatiter
np.flatnonzero
np.flexible
np.flip
np.fliplr
np.flipud
np.float
np.float16
np.float32
np.float64
np.float_
np.float_power
np.floating
np.floor
np.floor_divide
np.fmax
np.fmin
np.fmod
np.format_parser
np.frexp
np.frombuffer
np.fromfile
np.fromfunction
np.fromiter
np.frompyfunc
np.fromregex
np.fromstring
np.full
np.full_like
np.fv
np.generic
np.genfromtxt
np.geomspace
np.get_array_wrap
np.get_include
np.get_printoptions
np.getbufsize
np.geterr
np.geterrcall
np.geterrobj
np.gradient
np.greater
np.greater_equal
np.half
np.hamming
np.hanning
np.heaviside
np.histogram
np.histogram2d
np.histogramdd
np.hsplit
np.hstack
np.hypot
np.i0
np.identity
np.iinfo
np.imag
np.in1d
np.index_exp
np.indices
np.inexact
np.inf
np.info
np.infty
np.inner
np.insert
np.int
np.int0
np.int16
np.int32
np.int64
np.int8
np.int_
np.int_asbuffer
np.intc
np.integer
np.interp
np.intersect1d
np.intp
np.invert
np.ipmt
np.irr
np.is_busday
np.isclose
np.iscomplex
np.iscomplexobj
np.isfinite
np.isfortran
np.isin
np.isinf
np.isnan
np.isnat
np.isneginf
np.isposinf
np.isreal
np.isrealobj
np.isscalar
np.issctype
np.issubclass_
np.issubdtype
np.issubsctype
np.iterable
np.ix_
np.kaiser
np.kron
np.ldexp
np.left_shift
np.less
np.less_equal
np.lexsort
np.lib
np.linalg
np.linspace
np.little_endian
np.load
np.loads
np.loadtxt
np.log
np.log10
np.log1p
np.log2
np.logaddexp
np.logaddexp2
np.logical_and
np.logical_not
np.logical_or
np.logical_xor
np.logspace
np.long
np.longcomplex
np.longdouble
np.longfloat
np.longlong
np.lookfor
np.ma
np.mafromtxt
np.mask_indices
np.mat
np.math
np.matmul
np.matrix
np.matrixlib
np.max
np.maximum
np.maximum_sctype
np.may_share_memory
np.mean
np.median
np.memmap
np.meshgrid
np.mgrid
np.min
np.min_scalar_type
np.minimum
np.mintypecode
np.mirr
np.mod
np.modf
np.moveaxis
np.msort
np.multiply
np.nan
np.nan_to_num
np.nanargmax
np.nanargmin
np.nancumprod
np.nancumsum
np.nanmax
np.nanmean
np.nanmedian
np.nanmin
np.nanpercentile
np.nanprod
np.nanstd
np.nansum
np.nanvar
np.nbytes
np.ndarray
np.ndenumerate
np.ndfromtxt
np.ndim
np.ndindex
np.nditer
np.negative
np.nested_iters
np.newaxis
np.nextafter
np.nonzero
np.not_equal
np.nper
np.npv
np.numarray
np.number
np.obj2sctype
np.object
np.object0
np.object_
np.ogrid
np.oldnumeric
np.ones
np.ones_like
np.outer
np.packbits
np.pad
np.partition
np.percentile
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