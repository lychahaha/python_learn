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
