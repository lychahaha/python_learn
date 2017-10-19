import numpy as np

#全局改变
##改变形状
np.reshape(a, shape)
np.resize(a, shape)#size可以改变,变大时填充a
np.squeeze(a, axis=None)#去掉长度为1的维度,axis的长度需要等于1,为None时全去掉
np.expand_dims(a, axis)#插入一个长度为1的维度
np.ravel(a)#flat
##改变迭代方向
np.transpose(a, axes=None)
np.moveaxis(a, src, dst)#高级转置
np.swapaxes(a, axis1, axis2)
np.roll(a, shift, axis=None)#滚动,axis为None时按flat后的顺序滚,但shape不变
#np.rollaxis(a, axis, dst)#把某轴滚到某个维度
np.rot90(a, k=1, axes=(0,1))#数组旋转90度,k是旋转次数,axes是转轴
np.flip(a, axis)#reverse
np.fliplr(a)#np.flip(a, 1)
np.flipud(a)#np.flip(a, 0)

#局部改变
##插入与删除
np.insert(a, ixs, val, axis=None)
np.delete(a, ixs, axis=None)#删除axis下某些下标对应的行,没axis时会flat
np.append(a, b, axis=None)#按axis进行append,两数组维数应该一样,不给axis则先flat成一维
##连接与分割
np.concatenate(arrays, axis=0)
np.vstack(arrays)#np.concatenate(a, axis=0)
np.hstack(arrays)#np.concatenate(arrays, axis=1)
np.dstack(arrays)#np.concatenate(arrays, axis=2)
np.row_stack(arrays)#np.vstack
np.column_stack(arrays)#np.hstack
np.stack(arrays, axis=0)#相当于新维度插入到axis
np.split(a, n, axis=0)#这个n可以是数组,表示[beg:n[0]]...[n[i]:n[i+1]]...[n[-1]:end]
np.vsplit(a, n)#np.split(a, n, axis=0)
np.hsplit(a, n)#np.split(a, n, axis=1)
np.dsplit(a, n)#np.split(a, n, axis=2)
np.array_split(a, n, axis=0)#n可以不整除维度的长度
np.block(a)#类似分块矩阵的合并

#读写
np.load(file, encoding='ACSII')
np.loads(file, encoding='ACSII')
#np.loadtxt(file, dtype=np.float)
np.save(file, a)
np.savez(file, **kw)#存多个数组
np.savez_compressed(file, **kw)#带压缩
#np.savetxt
np.frombuffer(buf, dtype=np.float, count=-1, offset=0)
np.fromfile(file, dtype=np.float, count=-1, sep='')
np.fromfunction(func, shape)#func的参数是下标,返回值
np.fromiter(iter, dtype, count=-1)
np.fromstring(str, dtype=np.float, count=-1, sep='')

#聚类型运算
##数值
np.average(a, axis=None, weights=None)
np.sum(a)
np.var(a)#方差
np.std(a)
np.mean(a)#均值
np.median(a)#中位数
np.prod(a)
np.product(a)#np.prod
np.percentile(a, q)#求百分数
np.count_nonzero(a)
np.bincount(a, weights=None)#计算元素出现次数,a必须是1维非负整数数组
##逻辑
np.all(a)
np.alltrue(a)#np.all
np.any(a)
np.sometrue(a)#np.any
np.array_equal(a, b)#完全相等
np.array_equiv(a, b)#广播后相等
np.allclose(a,b)#比较两个数组的元素是否某种程度上接近
##直方图
np.histogram(a, bins=10, range=None, weights=None, density=None)#bins决定分组界线,range决定边界,weights决定加权,density决定是否归一化
#np.histogram2d
#np.histogramdd

#生成
##数列型生成
np.arange(start, stop, step)
np.geomspace(start, stop, num, endpoint=True)#生成几何级数,endpoint表明是否右闭
np.linspace(start, stop, num=50, endpoint=True, retstep=False)#生成算术级数(等差数列),retstep为true返回(数列,公差)
np.logspace(start, stop, num=50, endpoint=True, base=10.0)#对算术级数取指数
##填充型生成
np.empty(shape)
np.empty_like(a)
np.full(shape, x)
np.full_like(a, x)
np.ones(shape)
np.ones_like(a)
np.zeros(shape)
np.zeros_like(a)
##重复型生成
np.repeat(a, rep, axis=None)#沿着某个轴对每个元素重复rep次,rep为数组时表示每个元素的重复次数
np.tile(a, shape)#按shape平铺
np.broadcast_to(a, shape)
##特殊型
np.vander(a, N=None, increasing=False)#生成范德蒙德矩阵,N是列数,默认是len(a),increasing表示指数是否从左到右增长
np.identity(n)#单位矩阵

#下标
np.indices(shape)#生成分布下标
np.nonzero(a)#返回非0的分布下标
np.unravel_index(ixs, shape)#flat下标->分布下标
np.flatnonzero(a)#返回flat后的a中非0的元素下标
np.ndindex(*shape)#迭代器,生成全下标
np.ndenumerate(a)#迭代器,生成(全下标,值)

#下标读写
np.put(a, ix, x)#a.flat[ix]=x
np.putmask(a, mask, vals)#a.flat[i]=vals.flat[i] if mask.flat[i]
np.take(a, ixs, axis=None)#axis为None时,按flat索引

#map
np.apply_along_axis(func, axis, a)#沿着某维度apply
np.apply_over_axes(func, a, axes)#沿着多个维度apply

#条件
np.where(cond, [a,b])#true->a, false->b,如果没有[a,b],则返回np.nonzero(cond)
np.choose(a, c)#[c[a[i]][i] for i in a.size]
np.compress(conditions, a, axis=None)#根据条件在轴上进行过滤
np.piecewise(a, masklist, funclist)#相当于实现多重if的map
np.select(condlist, arraylist)#相当于多重if的map
np.extract(condition, a)#condition的shape和a的要相同,符合条件的会选出来变成1维数组,相当于a[b]

#创建
np.array(l)
np.asarray(a)
np.asfarray(a)#强制转成浮点数
np.asmatrix(a)
np.mat(obj)
np.matrix(obj)
np.asscalar(a)

#基础运算
##加减乘
np.add(a,b)#a+b
np.subtract(a, b)#a-b
np.multiply(a, b)#a*b
##除法
np.true_divide(a, b)#a/b
np.floor_divide(a, b)#a//b
np.divide(a, b)#a/b
np.mod(a, b)#a%b(符号跟除数)
np.divmod(a, b)#返回(a//b,a%b)
np.fmod(a, b)#取余(符号跟被除数)
np.remainder(a, b)#a%b
##指数与对数
np.power(a, b)#a**b(a为整数,b为负数时报错)
np.float_power(a, b)#强制转换成浮点数
np.exp(a)#e**a
np.expm1(a)#np.exp(a)-1
np.exp2(a)#2**a
np.log(a)#ln(x)
np.log10(a)#lg(x)
np.log1p(a)#ln(1+x)
np.log2(a)#log2(x)
np.logaddexp(a, b)#ln(e^a+a^b)
np.logaddexp2(a, b)#log2(2^a+2^b)
np.ldexp(a, b)#a*2**b
np.frexp(a)#返回(m,e),a=m*2**e,其中0<=m<1
##平方与立方
np.sqrt(a)
np.square(a)#a**2
np.hypot(a, b)#sqrt(a**2+b**2)
np.cbrt(a)#立方根
##三角函数
np.cos(a)
np.cosh(a)
np.sin(a)
np.sinc(a)#sin(x)/x
np.sinh(a)
np.tan(a)
np.tanh(a)
np.arccos(a)
np.arccosh(a)
np.arcsin(a)
np.arcsinh(a)
np.arctan(a)
np.arctan2(ay, ax)
np.arctanh(a)

#基础符号
##算术逻辑
np.equal(a, b)#a==b
np.not_equal(a, b)#a!=b
np.greater(a, b)#a>b
np.greater_equal(a ,b)#a>=b
np.less(a, b)#a<b
np.less_equal(a, b)#a<=b
np.left_shift(a, b)#a<<b
np.right_shift(a, b)#a>>b
np.positive(a)#+a
np.negative(a)#-a
##布尔逻辑
np.logical_and(a, b)
np.logical_not(a)
np.logical_or(a, b)
np.logical_xor(a, b)
np.bitwise_and(a, b)#a&b,按位与,输入必须是int或bool
np.bitwise_not(a)#~a
np.bitwise_or(a, b)#a|b
np.bitwise_xor(a, b)#a^b
np.invert(a)#np.bitwise_not
np.in1d(a, b, invert=False)#判断a中元素是否在b中,invert表示结果是否取反

#最大与最小
##聚类
np.max(a)#优先返回np.nan
np.min(a)
np.amax(a)#np.max
np.amin(a)#np.min
np.ptp(a)#np.max(a)-np.min(a)
##基础
np.maximum(a, b)#优先返回np.nan
np.minimum(a, b)
np.fmax(a, b)#优先不返回np.nan
np.fmin(a, b)

#arg-
#取结果的下标
np.argmax(a)
np.argmin(a)
#np.argpartition
np.argsort(a)
np.argwhere(a)#返回true的全下标

#nan-
#不考虑np.nan或把它当成透明
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

#is-
np.isclose(a, b)#是否接近
np.iscomplex(a)
np.iscomplexobj(a)#至少有一个复数
np.isreal(a)
np.isrealobj(a)#是否完全没有复数
np.isfinite(a)
np.isfortran(a)
np.isin(a, b, invert=False)#判断a中元素是否在b中,invert表示结果是否取反
np.isinf(a)
np.isnan(a)
np.isnat(a)#is not a time
np.isneginf(a)
np.isposinf(a)
np.isscalar(a)
np.issctype(a)#is scalar type
np.iterable(a)#是否能遍历

#取整
np.around(a, decimals=0)#四舍五入,decimals决定对几位小数进行四舍五入,decimals可以为负
np.floor(a)
np.ceil(a)
np.round(a, decimals=0)#np.around
np.round_(a, decimals=0)#np.around
np.rint(a)#返回离它最近的整数
np.trunc(a)#去掉小数部分
np.fix(a)#往0的方向取整
np.modf(a)#返回(小数部分,整数部分)

#绝对值与分段函数
np.abs(a)#(复数的话求长度)
np.absolute(a)#np.absolute
np.fabs(a)#强制转换为浮点数,不支持复数
np.clip(a, min_, max_)#截断,相当于a<min_ -> min_, min_<=a<=max_ -> a, a>max_ -> max_
np.sign(a)#a/|a|,相当于a<0 -> -1, a==0 -> 0, a>0 -> 1
np.heaviside(a, b)#a<0 -> 0, a==0 -> b, a>0 -> 1
np.copysign(a, b)#结合a的绝对值,b的正负号
np.signbit(a)#是否小于0

#集合运算
np.setdiff1d(a, b)#list(set(a-b))
np.setxor1d(a, b)#list(set(a^b))
np.union1d(a, b)#list(set(a||b))
np.intersect1d(a, b)#list(set(a&&b))

#微积分
np.diff(a, n=1)#n阶差分
np.ediff1d(a, to_end=None, to_begin=None)#a进行flat后求差分,并在首尾插入to_begin和to_end,它们可以是标量或数组
np.trapz(y, x=None, dx=1.0, axis=-1)#梯形积分,x为None时认为x为等差数列,公差为dx
np.gradient(a)#求梯度

#排序
np.sort(a)
np.msort(a)#np.sort(a, axis=0)
np.sort_complex(a)#key=(real,imag)
np.lexsort(arrays)#按keys进行sort然后返回下标

#二分
np.searchsorted(a, x)#二分查找
np.digitize(x, y, right=False)#x的每个元素在有序数组y中二分,返回在y中的下标,right为true表示下界,false表示上界

#积
np.inner(a, b)#内积
np.outer(a, b)#外积
np.kron(a, b)#kron积(张量积)
np.matmul(a, b)
np.dot(a, b)
np.tensordot#很难的dot
np.vdot(a, b)#np.sum(a.flat*b.flat)

#内存
np.copy(a)#深拷贝
np.may_share_memory(a, b)#是否占用共同的内存(只检查边界是否有交集)
np.shares_memory(a, b)#(检查是否有元素被共同)
np.byte_bounds(a)#返回(beg,end),该数组占用内存的起终指针

#前缀型计算
np.cumsum(a)#前缀和
np.cumprod(a)#前缀积
np.cumproduct(a)#np.cumprod

#矩阵对角
##对角线
np.diag(a)#输入什么输出另一个(对角线向量<->对角矩阵)
np.diag_indices(n, ndim=2)#生成对角线的分布下标
np.diag_indices_from(a)#生成该矩阵对角线的分布下标
np.diagflat(obj, k=0)#obj进行flat后,生成对角线矩阵,k是对角线偏移量,k=0表示obj是对角线
np.diagonal(a)#返回a的对角线,还有更复杂的功能
np.fill_diagonal(a, x)#x填入a的对角线
np.eye(n, M=None, k=0)#单位矩阵,n是行数,M是列数默认等于n,k是对角线偏移量
np.trace(a, offset=0, axis1=0, axis2=1)#矩阵的迹
##上三角下三角
np.tri(n, M=None, k=0)#下三角矩阵,n,m为行列数,k为偏移量
np.tril(a, k=0)#a取下三角矩阵
np.tril_indices(n, k=0, m=None)#下三角矩阵的分布下标
np.tril_indices_from(a, k=0)#a的下三角矩阵的分布下标
np.triu(a, k=0)#a取上三角矩阵
np.triu_indices(n, k=0, m=None)#上三角矩阵的分布下标
np.triu_indices_from(a, k=0)#a的上三角矩阵的分布下标

#复数
np.real(a)#实部
np.imag(a)#虚部
np.conj(a)#np.conjugate
np.conjugate(a)#共轭
np.angle(a, deg=False)#返回复数的角度,deg为true返回角度制,false返回弧度制

#多项式
#np.poly(a)#求出多项式系数,a是所有的零点
np.poly1d(a)#返回一个多项式类,a是多项式系数
np.polyadd(pa, pb)
np.polysub(pa, pb)
np.polymul(pa, pb)
np.polydiv(pa, pb)#返回(商式,余式)
np.polyder(p, m=1)#求m阶导数
np.polyint(p, m=1, k=None)#求m阶积分,k是常数项,默认全0
np.polyfit(x, y, deg)#多项式插值,返回多项式系数,deg是多项式的度
np.polyval(p, x)#多项式求值
np.roots(p)#返回多项式的根

#输出
##获取输出选项值
np.get_printoptions()
##设置输出选项值
np.set_printoptions(precision=8) #浮点数小数位数为8
np.set_printoptions(threshold=1000) #不省略打印的阈值
np.set_printoptions(edgeitems=3) #省略打印时开头和结尾打印的个数
np.set_printoptions(linewidth=75) #每行最大字符数
np.set_printoptions(suppress=False) #是否禁止使用科学计数法打印小浮点数
np.set_printoptions(nanstr='nan') #nan的输出
np.set_printoptions(infstr='inf') #inf的输出
np.set_printoptions(formatter={'float':float.__str__}) #设置浮点数输出格式

#日期
np.busday_count(date_beg, date_end, )#计算日期相差天数
#np.busday_offset
#np.busdaycalendar
#np.is_busday

#二进制
np.packbits(a, axis=None)#二进制->uint8形式pack
np.unpackbits(a, axis=None)#uint8->二进制形式unpack

#类型转换
np.can_cast(t1, t2)#判断是否能类型转换
np.common_type(*arrays)#返回运算后会变成的类型
np.maximum_sctype(obj)#返回这个数据可能的最高类型
np.min_scalar_type(obj)#返回这个数据可能的最低类型
np.obj2sctype(obj)#obj在np里的type
np.promote_types(type1, type2)#返回两个type转换的最小精度type
np.result_type(*array)#返回结果的类型

#浮点值
np.nextafter(a, b)#返回a向b方向的下一个浮点值
np.nan_to_num(a)#把非数字改成极端数字
np.real_if_close(a, tol=1000)#如果复数接近0,就返回实部,接近是指<tol*float_eps
np.spacing(a)#返回比a大的最大浮点值减去a

#金融
np.fv(rate, nper, pmt, pv)#终值
np.pv(rate, nper, pmt, fv=0)#现值
np.npv(rate, values)#净现值
np.pmt(rate, nper, pv, fv=0)#每期支付金额,(pmt=ppmt+ipmt)
np.ppmt(rate, per, nper, pv, fv=0)#每期支付金额之本金
np.ipmt(rate, per, nper, pv, fv=0)#每期支付金额之利息
np.nper(rate, pmt, pv, fv=0)#定期付款期数
np.rate(nper, pmt, pv, fv, guess=0.1, tol=1e-6, maxiter=100)#利率
np.irr(values)#内部收益率
np.mirr(values, finance_rate, reinvest_rate)#修正的内部收益率

#常数
np.e
np.euler_gamma
np.pi
np.nan
np.inf
np.little_endian#true,小端格式

#形状信息
np.alen(a)#相当于shape[0]
np.ndim(a)#维度数,a.shape.size
np.shape(a)
np.size(a)
np.rank(a)#np.ndim(a)

#信息
np.info(obj)#输出帮助信息,obj是对象或者字符串
np.iinfo(dtype)#整数的信息
np.finfo(dtype)#浮点数的上下限等信息
np.who()#查看有什么数组

#record
#np.recarray
#np.recfromcsv
#np.recfromtxt
#np.reciprocal
#np.record

#弧度与角度
np.deg2rad(a)
np.degrees(a)#np.rad2deg
np.rad2deg(a)
np.radians(a)#np.deg2rad

#python函数
np.division
np.print_function

#类型
##整数
np.bool#bool
np.bool8#np.bool_
np.bool_

np.int#int
np.int0#np.int64
np.int16
np.int32
np.int64
np.int8
np.int_#np.int32
np.intc#np.int32
np.integer
np.intp#np.int64

np.uint
np.uint0#np.uint64
np.uint16
np.uint32
np.uint64
np.uint8
np.uintc#np.uint32
np.uintp#np.uint64

##浮点数
np.float
np.float16
np.float32
np.float64
np.float_#np.float64
np.half#np.float16

##复数
np.complex
np.complex128
np.complex64
np.complex_#np.complex128
np.cdouble#np.complex128
np.cfloat#np.complex128
np.clongdouble#np.complex128
np.clongfloat#np.complex128
##字符串
np.str
np.str0#np.str_
np.str_
np.string_#np.bytes


np.byte#np.int8
np.bytes0#np.btypes_
np.bytes_


#np.complexfloating
np.csingle#np.complex64
np.datetime64
np.double#np.float64


np.long#np.int
np.longcomplex#np.complex128
np.longdouble#np.float64
np.longfloat#np.float64
np.longlong#np.int64
np.object
np.object0#np.object_
np.object_
np.short#np.int16
np.single#np.float32
np.singlecomplex#np.complex64

np.ubyte#np.uint8

np.ulonglong#np.uint64
np.unicode#str
np.unicode_#np.str_
np.ushort#np.uint16
np.void
np.void0#np.void0

#模块
np.char
np.compat
np.ctypeslib
np.lib
np.linalg
np.ma
np.math
np.matrixlib
np.polynomial
np.random
np.rec
np.sys
np.testing
np.version
np.warnings

#数组属性
a = np.array([1,2,3])
##属性
a.dtype#元素类型
a.itemsize#元素占字节数
a.nbytes#总的占用字节数
a.strides#(总元素个数,每个元素占字节数)
a.flags#储存属性
a.base#展平的a(同一块内存)
a.flat#展平迭代器
##函数
a.astype(dtype)
a.flatten()#展平的a(不同一块内存)
a.byteswap(inplace=False)#每个元素内部bit进行reverse
a.tobytes()#->btypes
a.tostring()#a.tobytes
a.tofile(file, sep='', format="%s")
a.tolist()#->list
a.dump(file)
a.dumps(file)
##
a.ctypes
a.data
##same
a.max,a.min,a.mean,a.all,a.any,a.prod,a.sum,a.std,a.var,a.ptp#聚类
a.argmax,a.argmin,a.argpartition,a.argsort#arg-
a.transpose,a.swapaxes,a.ravel#局部改变
a.choose,a.compress#条件
a.reshape,a.resize,a.squeeze#全局改变
a.searchsorted,a.sort#二分与排序
a.shape,a.size,a.ndim#形状信息
a.fill,a.repeat#生成
a.put,a.take#下标读写
a.diagonal,a.trace#矩阵
a.clip#绝对值
a.copy#内存
a.cumprod,a.cumsum#前缀
a.imag,a.real,a.conj,a.conjugate#复数
a.nonzero#下标
a.dot#积
a.round#取整
##tmp
a.getfield
a.item
a.itemset
a.newbyteorder
a.partition
a.setfield
a.setflags
a.view


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

#np.absolute_import
#np.add_docstring
#np.add_newdoc
#np.add_newdoc_ufunc
#np.add_newdocs
#np.array2string
#np.array_repr
#np.array_str
#np.asanyarray
#np.asarray_chkfinite
#np.ascontiguousarray
#np.asfortranarray
#np.atleast_1d
#np.atleast_2d
#np.atleast_3d
#np.bartlett
#np.base_repr
#np.bench
#np.binary_repr
#np.blackman
#np.bmat
#np.broadcast
#np.broadcast_arrays
#np.c_
#np.cast
#np.character
#np.chararray
#np.compare_chararrays
np.convolve(a, b)#1维卷积
#np.copyto
#np.core
#np.corrcoef#相似度
#np.correlate
#np.cov
#np.cross
#np.datetime_as_string
#np.datetime_data
#np.deprecate
#np.deprecate_with_doc
#np.disp
np.dtype
#np.einsum
#np.einsum_path
#np.emath
#np.errstate
#np.fastCopyAndTranspose
#np.fft(a)
#np.find_common_type
#np.flatiter
#np.flexible
#np.floating
#np.format_parser
#np.frompyfunc
#np.fromregex
#np.generic
#np.genfromtxt
#np.get_array_wrap
#np.get_include
#np.getbufsize
#np.geterr
#np.geterrcall
#np.geterrobj
#np.hamming
#np.hanning
#np.i0
#np.index_exp
#np.inexact
np.infty
#np.int_asbuffer
#np.interp
#np.issubclass_
#np.issubdtype
#np.issubsctype
#np.ix_
#np.kaiser
#np.lookfor
#np.mafromtxt
#np.mask_indices
#np.memmap
#np.meshgrid
#np.mgrid
#np.mintypecode
np.nbytes#dict,记录类型所占字节数
np.ndarray
#np.ndfromtxt
#np.nditer
#np.nested_iters
#np.newaxis
#np.numarray
#np.number
#np.ogrid
#np.oldnumeric
#np.pad
#np.partition
#np.pkgload
#np.place(a, mask, vals)
#np.r_
#np.ravel_multi_index
#np.require
#np.s_
#np.safe_eval
#np.sctype2char
np.sctypeDict#dict, str->class
#np.sctypeNA#dict, ?
np.sctypes#dict, str->list_class(大类)
#np.set_numeric_ops
#np.set_string_function
#np.setbufsize
#np.seterr
#np.seterrcall
#np.seterrobj
#np.show_config
np.signedinteger
#np.source
#np.test
np.timedelta64
np.tracemalloc_domain#389047
np.trim_zeros(a, trim='fb')#去掉前后的零,trim='fb'表示前后都去,'f'表示去掉前,'b'表示去掉后
np.typeDict#dict
np.typeNA#dict
np.typecodes#dict
np.typename(ch)#type:ch->str
np.ufunc
#np.unique
np.unsignedinteger
#np.unwrap
#np.vectorize