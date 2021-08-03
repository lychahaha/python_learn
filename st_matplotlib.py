#jh:func-base,key-type
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

fig,ax = plt.subplots()

# 折线图,直方图,柱状图,饼状图,散点图,文字,图片
ax.plot(xs, ys)
ax.hist(vals, bins=100)
ax.bar(xs, heights)
ax.pie(vals, labels=labels)
ax.scatter(xs, ys)
ax.text(x, y, 'haha')
ax.imshow(img)


ax.set_title('title')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()

ax.set_xlim(0, 3)
ax.set_ylim(1, 4)
ax.set_xticks([-2,0,2])
ax.set_yticks([-1,0,1])
ax.set_xticklabels(['2006','2008','2010'])
ax.set_yticklabels(['low','medium','high'])
ax.set_aspect('equal')
ax.set_yscale('log')
ax.grid()

plt.show()
fig.savefig('a.png')



# figure, ax
# figure是一个图, ax是figure里的一个子图
# 一次show可以创建多个窗口, 一个窗口就是一个figure
# 一个figure里包含多个ax
# 一个ax里包含多条曲线,或其他东西


#  创建ax
## 当row和col都是1时, 返回一个ax对象
## 当row和col一个大于1, 一个等于1时, 返回一个ax的np一维数组
## 当row和col都大于1时, 返回一个ax的np二维数组
fig,ax = plt.subplots() #row和col的默认值是1
fig,axes = plt.subplots(2,1) #2行1列, 共2个ax
fig,axes = plt.subplots(2,3) #2行3列, 共6个ax


# plot(折线图)
## 返回值是list of line, 一般情况是list里只有一个line对象
ax.plot(xvals, yvals) #画
ax.plot(yvals) #xvals默认为[0,1,2...]
ax.plot(x1, y1, 'r', x2, y2, 'b') #多条线段一起画
ax.plot(xvals, yvals, label='line1') #设置图例里该线的标题
ax.plot(xvals, yvals, s) #设置样式
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

ax.plot(**kw) #各种各样的参数
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


# hist(直方图)
## 返回值是(cnts,intervals,Patches)
## 含义是每个区间的数目,区间的分界线(比区间数多1),每个区间的Patch对象
## 如果有多个数据,则第一个返回值是list(cnts),第三个返回值是list(Patches)
ax.hist(data) #直方图,data是一维数组
ax.hist([data1,data2]) #画多个数据的直方图
ax.hist(data, bins=50) #设置区间数
ax.hist(data, bins=list) #设置区间的分界线
## 其他参数
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
## kw(Patch需要的参数)


# bar(柱状图)
ax.bar(xs, heights)
ax.bar([0,1,2], [15,10,20], width=0.5, color='black', align='edge', label='bar', tick_label=['a','b','c'])
## 水平柱状图
### x变成bottom(y), height变成width
ax.bar(x=0, bottom=[0,1,2], height=0.5, width=[15,10,20], orientation='horizontal')
### 另一个api
ax.barh([0,1,2], [15,10,20], height=0.5)
## 画多个柱状图
### 主要是对齐问题
ax.bar(np.arange(3)-0.3, [5,3,2], width=0.3, align='edge')
ax.bar(np.arange(3), [1,4,6], width=0.3, align='edge')
## 叠加柱状图
ax.bar([0,1,2], h1)
ax.bar([0,1,2], h2, bottom=h1)
## 在柱子顶部添加数据
### 使用ax.text暴力实现
for a, b in zip(x, heights):
    plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
'''
bottom: float|list[float], 柱子底部起点, 默认是0
width: float|list[float], 柱子的宽度, 默认0.8
color: color, 柱子的颜色
edgecolor: color, 柱子边缘的颜色
align: 'center'(默认)|'edge', 设置x坐标是柱子的中心或左端点
label: str, 设置图例里的名字
tick_label: list[str], x坐标替换成自定义的文字
'''


# pie(饼状图)
ax.pie([2,3,4], labels=['CN','KR','JP']) #[2,3,4]是比例
ax.pie([2,3,4], labels['C','K','J'], colors=['r','g','b'], explode=[0.2,0,0.1], autopct='%.1f%%', pctdistance=0.8, shadow=True)
'''
colors: list[color], 每种饼的颜色
explode: list[float], 每种饼离中心的距离(0~1)
autopct: format str, 如'%.1f%%', 有这个参数则打印出每个饼的比例
pctdistance: float, 每种饼的比例文字离中心的距离(0~1)
shadow:bool, 是否加阴影, 显得更立体
center: (float,float), 设置饼的圆心坐标
radius: float, 饼的半径
startangle: float(0~360), 设置角度起点, 默认0度是3点钟方向, 逆时针
frame: bool, 是否画坐标轴和框之类的,默认False
'''


# scatter(散点图)
ax.scatter(xvals, yvals)
ax.scatter(xvals, yvals, s=sizes, c=colors, edgecolor='black', facecolor='r')
'''
s: list[float], 每个点画出来的圆的面积
c: color|list[color], 每个点的颜色
edgecolor: color|list[color], 每个点的边缘颜色(设置此参数会屏蔽color参数)
facecolor: color|list[color], 每个点的内部颜色(设置此参数会屏蔽color参数)
'''


# text(文字)
## 返回值是Text对象
## 字体不能影响轴的lim
ax.text(x, y, s)
ax.text(x, y, s, withdash=False)#?
ax.text(x, y, s, fontdict=None)#通过字典设置字体
ax.text(x, y, s, fontsize=12)#字典中的各种属性也可以直接放在参数表上



# image(图片)
ax.imshow(img)
'''
img: np数组(h,w)|(h,w,3)|(h,w,4), float|uint8
'''


# patch
ax.add_patch(p)
p.set_zorder(1) #设置z轴

## polygon
poly = plt.Polygon([(0,0),(1,0),(1,1),(0,1),(0,0)])

# title,xlabel,ylabel
## 返回值是Text对象
ax.set_title(s)
ax.set_xlabel(s)
ax.set_ylabel(s)
ax.set_title(s, fontdict=None) #参考plt.text
ax.set_title(s, loc='center') #标题对齐,'center'|'left'|'right'


# legend
## 返回值是Legend对象
ax.legend() #设置显示图例
ax.legend([line1,line2], ['one','two']) #对某些对象设置图例
ax.legend(loc='upper left', bbox_to_anchor=(0.1,0.1), ncol=2, handletextpad=0.0)
'''
loc:'best','right','center','upper left','lower right','center right','left center'等等的各种组合
bbox_to_anchor: 基于loc的偏置微调
ncol:列数
handletextpad:图标和文字的间隔
'''
ax.legend(prop={'family':'Times New Roman'})


# xlim,ylim
## 返回值是(min,max)
ax.set_xlim() #返回[xmin,xmax]
ax.set_xlim([xmin,xmax]) #设置xmin和xmax
ax.set_xlim(xmin, xmax)
ax.set_xlim(xmin=xmin)
ax.set_xlim(xmax=xmax)
ax.set_ylim([ymin,ymax])
# axis
## 返回值是[xmin,xmax,ymin,ymax]
ax.axis() #为了得到[xmin,xmax,ymin,ymax]
ax.axis([xmin,xmax,ymin,ymax]) #设置这些属性
ax.axis(s) #设置样式
'''
'off':去掉坐标轴
'equal':使x轴和y轴比例尺一样
'scaled':使x轴和y轴比例尺一样,但是窗口会被裁剪
'tight':?
'image':?
'square':使比例尺和轴的总长度都一样
'''


# tick
## 设置要显示的刻度
ax.set_xticks([-1,0,1]) 
ax.set_yticks([-1,0,1])
## 设置刻度的内容
ax.set_xticklabels(['a','b'])
ax.set_yticklabels(['a','b'])


# aspect
## 设置y单位向量与x单位向量的比例
ax.set_aspect('equal')
'''
aspect: 'auto'(默认)|'equal'|float
'''


# xscale,yscale
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


# grid
ax.grid(True) #设置显示坐标网格
## 各种参数
'''
which:'major'(默认)|'minor'|'both',?
axis:'both'(默认)|'x'|'y',哪个轴画网线
'''
## kw,参考plot中的kw


# save
fig.savefig('a.png')






# annotate(箭头)
## 返回值是Annotation对象
ax.annotate('text', xy=(2,3), xytext=(4,6), arrowprops=dict(facecolor='black',shrink=0.05))
## 参数
'''
xy:(x,y),箭头尖尖的坐标
xytext:(x,y),文字坐标(箭头底部坐标),不提供则把文字画在xy处,此时箭头会变成一个三角标
arrowprops:dict,箭头属性,不提供则不画箭头
{
    facecolor:箭头颜色
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


#中文问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False



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








#tight_layout
plt.tight_layout()#调整子图的间距
#参数
'''
pad:float,默认1.08,所有的间距?
h_pad,w_pad:float,垂直和水平的子图间距
rect:(left,bottom,right,top),?
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




#其他
##去掉边框
ax.spines['top'].set_visible(False) #top|bottom|left|right
