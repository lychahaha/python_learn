#coding=utf-8
import numpy
import matplotlib
from matplotlib import pyplot

#中文问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

'''
#简单画函数
xvals = numpy.linspace(0, 2*numpy.pi, 50)
yvals = numpy.sin(xvals)
pyplot.plot(xvals, yvals, linewidth=1.0)#第三个参数是线条粗细

pyplot.ylim(-2,2)#设置x轴范围
pyplot.ylim(-2,2)#设置y轴范围

pyplot.xlabel('x')
pyplot.ylabel('sin(x)')
pyplot.title(u'简单画函数')

pyplot.grid(True)

#pyplot.savefig('sin.png')

pyplot.show()
'''

'''
#图与子图
pyplot.figure(1)#创建(选择)图1
pyplot.figure(2)
ax1 = pyplot.subplot(211)#在图2创建子图1
ax2 = pyplot.subplot(212)#在图2创建子图2

xvals = numpy.linspace(0, 5, 5)

pyplot.figure(1)#选择图1
pyplot.plot(xvals, numpy.exp(xvals))
pyplot.title(u'exp函数')

pyplot.sca(ax1)#选择图2的子图1
pyplot.plot(xvals, numpy.sin(xvals))
pyplot.title(u'sin函数')

pyplot.sca(ax2)#选择图2的子图2
pyplot.plot(xvals, numpy.cos(xvals))
pyplot.title(u'cos函数')

pyplot.show()
'''

'''
#散点图
xvals = numpy.linspace(0,10,20)
yvals = numpy.sqrt(xvals)
pyplot.plot(xvals, yvals, ':r')#'o'代表点,'r'是红色而已
pyplot.show()
#其他样式
#颜色:b:blue,g:green,r:red,c:cyan,m:magenta,y:yellow,k:black,w:white
#线条:'-':solid实线,'--':虚线,'-.':虚线+点,':':点
#描点:'o':圆,'s':正方形,'p':五边形,'*':星形,'h':竖六边形,'H':横六边形,'+':加号,'x':x形,'D':菱形,'d':尖菱形
'''

'''
#画多条线
xvals = numpy.linspace(0,1,100)
yvals1 = numpy.sqrt(xvals)
yvals2 = numpy.exp(xvals)
pyplot.plot(xvals, yvals1)
pyplot.plot(xvals, yvals2)
pyplot.show()
'''

'''
#图例
xvals = numpy.linspace(0,1,100)
yvals1 = numpy.sqrt(xvals)
yvals2 = numpy.exp(xvals)
plot1, = pyplot.plot(xvals, yvals1, 'r')#注意逗号
plot2, = pyplot.plot(xvals, yvals2, 'g')
pyplot.legend([plot1,plot2], ('red','green'))
pyplot.show()
'''

'''
#直方图
data = numpy.random.normal(0, 1, 1000)#正态分布(均值,方差),第三个参数表示产生多少个随机数
pyplot.hist(data, histtype='stepfilled')#histtype参数去掉了内边框
pyplot.show()
'''

'''
#直方图自定义区间
data = numpy.random.normal(0, 1, 1000)
bins = numpy.arange(-4,4,0.5)
pyplot.hist(data, bins, histtype='stepfilled')
pyplot.show()
'''

#3d
'''
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


fig = plt.figure('xxx')
#面
ax = fig.add_subplot(1,2,1, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.7, cmap='jet', rstride=1, cstride=1, lw=0)
#线框
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_wireframe(X, Y, Z, rstride=3, cstride=3, lw=0.5)

plt.show()
'''

##
from matplotlib import pyplot as plt
import numpy as np

plt.show()

#plot
#返回值是list of line
plt.plot(xvals, yvals)#画
plt.plot(yvals)#xvals默认为[0,1,2...]
plt.plot(xvals, yvals, s)#设置样式
'''
#不同类型样式可以组合,比如'ro'
#颜色
b:blue,g:green,r:red,c:cyan,m:magenta,y:yellow,k:black,w:white
#线条
'-':solid实线,'--':虚线,'-.':虚线+点,':':点,' ':不画
#描点
'.':点,',':像素点,'o':圆,'s':正方形,'+':加号,'x':x形,'*':星形
'v','^','<','>':不同方向的三角形,'1','2','3','4':不同方向的菱角
'p':五边形,'h':竖六边形,'H':横六边形,'D':菱形,'d':尖菱形
'_','|':横竖的线
'''
plt.plot(x1, y1, 'r', x2, y2, 'b')#多条线段一起画
plt.plot(**kw)#各种各样的参数
'''
alpha:float,透明度
animated:bool?
antialiased(aa):bool?
axes:class(Axes),轴的属性
clip_box:class(BBox)?
clip_on:bool?
clip_path:?
color(c):str(名字或#030FB2这种表示)|1,2,3,4个float的list(1个不用list)
contains:?
dash_capstyle:'butt'|'round'|'projecting',?
dash_joinstyle:'miter'|'round'|'bevel',?
dashes:?
drawstyle:'full'|'left'|'right'|'bottom'|'top'|'none',?
figure:class(Figure),?
fillstyle:'full'|'left'|'right'|'bottom'|'top'|'none'
gid:str,?
label:str,标题
linestype(ls):'solid','-'|'dashed','--'|'dashdot','-.'|'dotted',':'|(offset,on-ofdash-seq)|'None',' ',''
linewidth(lw):float,线宽
marker:?
markeredgecolor(mec):?
markeredgewidth(mew):float,?
markerfacecolor(mfc):?
markerfacecoloralt(mfcalt):?
markersize(ms):float,?
markevery:?
picker:float,?
pickradius:float,?
rasterized:bool|None,?
solid_capstyle:'butt'|'round'|'projecting',?
solid_joinstyle:'miter'|'round'|'bevel',?
transform:class(Transform)?
url:str,?
visible:bool?
xdata:1D array,?
ydata:1D array,?
zorder:?
'''

#hist
#返回值是(cnts,intervals,Patches)
#含义是每个区间的数目,区间的分界线(比区间数多1),每个区间的Patch对象
#如果有多个数据,则第一个返回值是list(cnts),第三个返回值是list(Patches)
plt.hist(data)#直方图,data是一维数组
plt.hist([data1,data2])#画多个数据的直方图
plt.hist(data, bins=50)#设置区间数
plt.hist(data, bins=list)#设置区间的分界线
#其他参数
'''
range:(xmin,xmax),设置区间边界(bins为区间分界线时无效)
normed:bool,设置是否归一化(归一化后sum(区间cnt*区间长度)==1)
weights:array,设置加权,shape要和x一样
cumulative:bool,是否设置成data的前缀和的直方图
bottom:ymin,设置y轴起点(如为array,则是每个bin单独设下限,shape要和x一样)
histtype:'bar'(默认,柱状)|'barstacked'(多个数据会堆叠)|'step'(变成连续的线框)|'stepfilled(连续线框并填充)'
align:'left'|'mid'|'right',是否稍稍偏左或偏右
orientation:'horizontal'(水平放置)|'vertical'(默认)
rwidth:float,(0,1],设置宽度的缩小比例,使柱子变细并且不连续,'step'或'stepfilled'时无效
log:bool,设置y轴是否取log,取log会把cnt==0的柱子删除,返回值上也不会出现
color:柱状图的颜色
label:
stacked:bool,是否堆叠,效果和'barstacked'一样,但histtype可以设置成'step'使得效果叠加
'''
#kw(Patch需要的参数)

#axis
#返回值是[xmin,xmax,ymin,ymax]
plt.axis()#为了得到[xmin,xmax,ymin,ymax]
plt.axis([xmin,xmax,ymin,ymax])#设置这些属性
plt.axis(s)#设置样式
'''
'off':去掉坐标轴
'equal':使x轴和y轴比例尺一样
'scaled':使x轴和y轴比例尺一样,但是窗口会被裁剪
'tight':?
'image':?
'square':使比例尺和轴的总长度都一样
'''

#xlim,ylim
#返回值是(min,max)
plt.xlim()#返回[xmin,xmax]
plt.xlim([xmin,xmax])#设置xmin和xmax
plt.xlim(xmin, xmax)
plt.xlim(xmin=xmin)
plt.xlim(xmax=xmax)
plt.ylim([ymin,ymax])

#xlabel,ylabel
#返回值是Text对象
plt.xlabel(s)#设置x轴的标签
plt.xlabel(s, fontdict=None)#参考plt.text
plt.ylabel(s)

#text
#返回值是Text对象
#字体不能影响轴的lim
plt.text(x, y, s)
plt.text(x, y, s, withdash=False)#?
plt.text(x, y, s, fontdict=None)#通过字典设置字体
plt.text(x, y, s, fontsize=12)#字典中的各种属性也可以直接放在参数表上
'''
'''

#setp
plt.setp(line, color='r', linewidth=2.0)#设置属性
plt.setp(line, 'color', 'r', 'linewidth', 2.0)#matlab参数形式
plt.setp(lines, color='r')#设置多条线段

#title
#返回值是Text对象
plt.title(s)#设置标题
plt.title(s, fontdict=None)#参考plt.text
plt.title(s, loc='center')#标题对齐,'center'|'left'|'right'

#figure
#返回值是Figure对象
plt.figure()#创建新的匿名figure(第一个默认的不创建)
plt.figure(x)#(创建并)切换到id为x的figure
#各种参数
'''
num:int|str,figure的id,默认None,为str的话title会被设置成它
figsize:[int,int],窗口的宽高,默认None
dpi:int,分辨率,默认None
facecolor:背景颜色,默认None
edgecolor:边框颜色,默认None
'''

#subplot
#返回值是AxesSubplot
#该函数用于定义当前figure的行列数,以及切换当前子图
plt.subplot(nrows,ncols,plot_number)#行数,列数,当前子图(行号*ncols+列号+1)
plt.subplot(221)#快速写法
#各种参数
'''
facecolor:背景颜色
polar:bool,是否使用极坐标,默认False
projection:str,?
'''

#subplot2grid
#返回值是AxesSubplot
plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=2)#多行多列的子图
plt.subplot2grid((2,2), (0,0))#相当于plt.subplot(2,2,1)

#grid
plt.grid(True)#设置显示坐标网格
#各种参数
'''
which:'major'(默认)|'minor'|'both',?
axis:'both'(默认)|'x'|'y',哪个轴画网线
'''
#kw,参考plot中的kw

#savefig
plt.savefig(file)#保存figure
#各种参数

#sca
plt.sca(ax)#设置当前的子图

#gca
plt.gca()#获取当前的子图

#legend
#返回值是Legend对象
plt.legend([line1,line2], ['one','two'])#设置图例
#各种参数

#annotate
#返回值是Annotation对象
plt.annotate('text', xy=(2,3), xytext=(4,6), arrowprops=dict(shrink=0.05))#画箭头
#参数
'''
xy:(x,y),箭头尖尖的坐标
xytext:(x,y),文字坐标(箭头底部坐标),不提供则把文字画在xy处,此时箭头会变成一个三角标
arrowprops:dict,箭头属性,不提供则不画箭头
{
    width:箭头身体宽度
    headwidth:箭头尖尖宽度
    headlength:箭头尖尖长度
    shrink:不画的比例,就是箭头实际底部与xytext的距离占xy到xytext的比例
    ?:还有
}
xycoords:?
textcoords:?
annotation_clip:bool,如果true,则xy在原本的区域里才会被画,默认None,当xycoords=='data'时认为是true
'''

#xscale,yscale
plt.xscale('log')#改变坐标轴的量化
plt.yscale('log')
#scale(不同的scale可以使用不同的参数)
'''
'linear'(线性,默认)

'log'(对数)
{
    basex,basey:?
    nonposx,nonposy:'mask'|'clip',?
    subsx,subsy:?
}

'logit'(对数0到1)
{
    nonpos:'mask'|'clip',?
}

'symlog'(有负的对数)
{
    basex,basey:?
    linthreshx,linthreshy:?
    subsx,subsy:?
    linscalex,linscaley:?
}
'''

#tight_layout
plt.tight_layout()#调整子图的间距
#参数
'''
pad:float,默认1.08,所有的间距?
h_pad,w_pad:float,垂直和水平的子图间距
rect:(left,bottom,right,top),?
'''
