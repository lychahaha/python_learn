import torch

import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init

import torch.autograd as autograd 
import torch.optim as optim
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from collections import OrderedDict

#层定义
conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=1) #[5,5,1,6]
fc1 = nn.Linear(84, 10)
drop1 = nn.Dropout(p=0.5)
relu1 = nn.ReLU(True)
embed = nn.Embedding(2, 5)
#forward
x = conv1(x)
x = fc1(x)
x = F.max_pool2d(x, (2,2))
x = x.view(-1, 500) #flatten
x = F.relu(x)
x = drop1(x)
x = F.log_softmax(x)

#训练
model.train()
output = model(input)
loss = criterion(output, label)
opt.zero_grad()
loss.backward()
opt.step()
#测试
model.eval()
output = model(input)


#全局改变
##改变形状
y = torch.squeeze(x, dim=None) #(共享内存)
y = torch.unsqueeze(x, dim) #expand_dim
ys = torch.chunk(x, chunk_size, dim=0) #?
y = torch.reshape(x, (2,3)) #重新分配内存
y = x.view(3, 4) #不重新分配内存
##转置
y = x.t() #0,1维转置
y = torch.transpose(x, dim0, dim1)
y = x.permute(2,1,0) #多个维度参与的transpose
y = x.T #维度完全翻转,如(2,3,5)->(5,3,2)
##连接与分割
x = torch.cat(inputs, dim=0) #concat
y = torch.stack(inputs, dim=0)
ys = torch.unbind(x, dim=0) #unstack
ys = torch.split(x, split_size, dim=0) #?
##填充(把长度为1的dim扩充)
y = x.expand(-1, 4) #(3,1)->(3,4),这里-1表示不变, 不重新分配内存
y = x.expand_as(x2)
y = x.repeat(*sizes) #重新分配内存,?

#下标
y = torch.gather(x, dim, index) #y[i][j][k] = x[i][index[i][j][k]][k] if dim=1
y = torch.index_select(x, dim, index) #index是一维数组
y = torch.masked_select(x, mask) #等于x[mask], x和mask的shape要一样
y = torch.nonzero(x) #返回的shape是(none,x_dims)
#切片
y = x.narrow(dim, beg, len) #x[...,beg:beg+len,...] 中间某一维切片,但切片可能无法表示dim
x.put(index, val) #index和val都是一维,把x当成一维处理x[index]=val

#生成
##随机
x = torch.rand(5, 3) #01均匀分布
x = torch.randn(5, 3) #标准正态分布
x = torch.randperm(n) #全排列
##填充
x = torch.empty(5, 3)
x = torch.ones(5, 3)
x = torch.zeros(5, 3)
x = torch.full((5,3), 2)
y = torch.empty_like(x)
y = torch.ones_like(x)
y = torch.zeros_like(x)
y = torch.full_like(x, 2)
##数列
x = torch.arange(start, end, step=1)
x = torch.linspace(start, end, steps=100) #左闭右闭
x = torch.logspace(start, end, steps=100)
#矩阵
x = torch.eye(n, m=None)
y = torch.diag(x, diagonal=0) #矩阵<->对角线,diagonal控制对角线(大于0向上偏移)


#保存读取
torch.save(model.state_dict(), 'params.pth.tar') #只保存神经网络的模型参数
model.load_state_dict(torch.load('params.pth.tar'))
torch.save(model, 'model.pth.tar') #保存整个神经网络的结构和模型参数
model = torch.load('model.pth.tar')
model = torch.load('model.pth.tar', map_location='cpu') #以cpu模式打开


#tensor
##构造
x = torch.tensor(data, dtype=torch.float64, device='cuda:1', requires_grad=True)
x = torch.Tensor(5, 3)
x = torch.FloatTensor([0.1,0.2,0.3])
x = torch.cuda.FloatTensor([1,2,3])
##成员
fn = x.grad_fn #梯度函数
grad = x.grad #梯度
##属性
a = x.dtype #数据类型
a = x.shape
a = x.nelement() #size
a = x.ndim #维数
a = x.requires_grad #是否需要并回传梯度
a = x.is_leaf #是否是叶子
a = x.is_contiguous() #是否连续
a = x.is_cuda
##特殊方法
x.fill_(1.2)
x.map_(y, fx) #def fx:(a,b)->c

x.resize_(*sizes) #?
x.scatter(y, dim, index) #?
x.select(dim, index) #?
##
x.requires_grad_(True) #需要梯度
x.retain_grad() #非叶子节点也保留梯度
x.detach_() #将x从计算图中排除
##
x.index_add_?
x.index_copy_?
x.index_fill_?
x.index_put_?
x.masked_fill?
x.masked_scatter?
##复制
x.copy_(y) #将y的数值复制给x
y = x.detach() #没梯度的复制
y = x.clone() #梯度会回传到原tensor
y = x.contiguous() #如果内存连续,则返回x本身(一般切片和转置会导致不连续)
##类型转换
y = x.to(torch.int64)
y = x.to(torch.device('cuda'))
y = x.byte() #还有float,int
y = x.cuda()
##数据类型
torch.float,torch.float32
torch.double,torch.float64
torch.half,torch.float16
torch.uint8
torch.int8
torch.short,torch.int16
torch.int,torch.int32
torch.long,torch.int64
##tensor类型
torch.FloatTensor #默认
torch.DoubleTensor
torch.cuda.HalfTensor #16位浮点数
torch.ByteTensor #unsigned
torch.CharTensor
torch.ShortTensor
torch.IntTensor
torch.LongTensor
##设备类型
cuda1 = torch.device('cuda:1')
cpu = torch.device('cpu')
##库转换
x = x.numpy()
x = torch.from_numpy(x)
x = x.item() #0维张量变成python数字
x = x.tolist() #变成python的list



#cuda
a = torch.cuda.is_available() #是否能用gpu
a = torch.cuda.device_count()
torch.cuda.empty_cache() #清空缓存
torch.cuda.device(1) #切换到gpu-1
model = nn.DataParallel(model) #多GPU
#cpu
a = torch.get_num_threads()
torch.set_num_threads(a)

y = torch.bernoulli(x) #x<0.5->y=0,x>=0.5->y=1(x的值域必须是[0,1])
y = torch.multinomial(x, num_samples, replacement=False) #?
x = torch.normal(means, stds) #?


#测速上下文
with autograd.profiler.profile(enabled=True, use_cuda=False) as prof:
    pass
print(prof)
#报错显示正确错误位置的上下文
autograd.detect_anomaly()
#不计算梯度的上下文
torch.no_grad()



#数学
##分段函数
y = torch.abs(x)
y = torch.floor(x)
y = torch.ceil(x)
y = torch.round(x)
y = torch.clamp(x, min=None, max=None) #clip
y = torch.sign(x)
##整数小数
y = torch.trunc(x) #返回整数部分
y = torch.frac(x) #返回小数部分
##指数
y = torch.exp(x)
y = torch.log(x)
y = torch.log2(x)
y = torch.log10(x)
y = torch.log1p(x) #log(x+1)
##三角函数
y = torch.cos(x)
y = torch.sin(x)
y = torch.tan(x)
y = torch.cosh(x)
y = torch.sinh(x)
y = torch.tanh(x)
y = torch.acos(x)
y = torch.asin(x)
y = torch.atan(x)
y = torch.atan2(x, y)
##四则运算
z = torch.add(x, y) #?
torch.add(x, y, out=z)
z = y.add(x)
y.add_(x) #inplace
z = torch.mul(x, y)
z = torch.div(x, y)
z = torch.fmod(x, y)
z = torch.remainder(x, y)
y = torch.neg(x)
z = torch.pow(x, y)
##组合四则运算
x = torch.addcdiv(t, v, t1, t2) #t+v*(t1/t2) (v是标量)
x = torch.addcmul(t, v, t1, t2) #t+v*(t1*t2)
##特殊
y = torch.reciprocal(x) #1/x
y = torch.sqrt(x)
y = torch.rsqrt(x) #1/x^2
z = torch.lerp(x, y, w) #x+w(y-x) (w是标量)
##高级特殊
y = torch.sigmoid(x)
y = torch.dist(x, y, p=2) #x-y的p范数
##前缀
y = torch.cumsum(x, dim)
y = torch.cumprod(x, dim)
##布尔值
z = torch.eq(x, y)
z = torch.ge(x, y)
z = torch.gt(x, y)
z = torch.le(x, y)
z = torch.lt(x, y)
z = torch.ne(x, y)
y = x.all() #byteTensor独有
y = x.any() #byteTensor独有
##最大最小
z = torch.max(x, y)
y,y_ix = torch.max(x, dim)
z = torch.min(x, y)
##积
a = torch.dot(x, y)
z = torch.cross(x, y, dim=-1) #向量积?
z = torch.ger(x, y) #张量积?
##矩阵
z = torch.mm(x, y)
z = torch.mv(x, y) #mat*vec
a = torch.trace(x)
y = torch.tril(x, k=0) #下三角矩阵,k控制对角线
y = torch.triu(x, k=0) #上三角矩阵
vals,vecs = torch.eig(x, eigenvectors=False) #求特征值,eigenvectors表示是否计算特征向量
torch.gels #最小二乘法?
x,LU = torch.gesv(B, A) #解Ax=B
y = torch.inverse(x)
q,r = torch.qr(x)
s,v,d = torch.svd(x, some=True) #some?
##聚合
a = torch.mean(x)
y = torch.mean(x, dim)
y = torch.mean(x, dim, keepdim)
a = x.mean()
a = torch.var(x)
a = torch.std(x)
a = torch.sum(x)
a = torch.prod(x)
a = torch.max(x)
a = torch.min(x)
a = torch.norm(x, p=2) #p范数
a = torch.equal(x, y)


#算法
y,y_ix = torch.kthvalue(x, k, dim=None)
y,y_ix = torch.sort(x, dim=None, descending=False)
y,y_ix = torch.topk(x, k, dim=None, largest=True, sorted=True) #sorted表示输出是否排序
y,y_ix = x.median(x, dim=-1) #?
y,y_ix = x.mode(x, dim=-1) #众数?


y = torch.histc(x, bins=100, min=0, max=0) #如果min=max=0,则以x的min和max为界

y = torch.renorm(x, p, dim, maxnorm) #?

a = torch.is_tensor(x)





x.backward()
x.backward(k)
x.backward(retain_graph=True) #保留计算图,不然只能backward一次


#随机数种子
torch.manual_seed(seed)
a = torch.initial_seed() #返回当前种子
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


#optimizer
##构造
opt = optim.SGD(params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
opt = optim.Adam(params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0)
opt = optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
opt = optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0)
opt = optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0)
opt = optim.Adamax(params, lr=2e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0)
opt = optim.ASGD(params, lr=0.01, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0)
opt = optim.LBFGS(params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
opt = optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
###不同参数不同超参数
opt = optim.SGD(
    [
        {'params':model.base.parameters()},
        {'params':model.classifier.parameters(), 'lr':1e-3}
    ],
    lr=1e-2, momentum=0.9
)

##方法
optimizer.step()
optimizer.zero_grad()
##读写
'''
state_dict包含state和param_groups,
state放着opt中间变量,比如adam的滑动均值和方差
param_groups主要是构造时给的东西
'''
optimizer.load_state_dict(state_dict)
d = optimizer.state_dict()

##学习率调度器
###构造
lrs = optim.lr_scheduler.LambdaLR(opt, lr_lambda, last_epoch=-1) #lr_lambda是epoch->k, lr=base_lr*k, 这里lr_lambda可以是针对多个group的list
lrs = optim.lr_scheduler.StepLR(opt, step_size, gamma=0.1, last_epoch=-1) #每step个epoch,学习率乘gamma
lrs = optim.lr_scheduler.MultiStepLR(opt, milestones, gamma=0.1, last_epoch=-1) #milestones是个step的列表,每到其中的某个step,学习率乘gamma
lrs = optim.lr_scheduler.ExponentialLR(opt, gamma, last_epoch=-1) #指数衰减,相当于step=1的stepLR
lrs = optim.lr_scheduler.CosineAnnealingLR(opt, T_max, eta_min=0, last_epoch=-1) #余弦衰减,lr=eta_min+(base_lr-eta_min)*(1+cos(t/T_max*pi))/2
lrs = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
'''
loss不变,才变学习率的调度器,变是指数衰减
mode:'min'|'max',表明loss是要尽量大还是尽量小
factor:相当于指数衰减里的gamma
patience:最多容忍多少个epoch
verbose:学习率改变时,是否打印
threshold:阈值
threshold_mode:'rel'|'abs',判断好坏是根据相对距离,还是绝对距离
cooldown:改变学习率后,有多少个epoch完全不管好坏
min_lr:学习率下限
'''
###方法
lrs.step(epoch=None) #更新epoch,默认+1
lrs.get_lr() #当前lr
###读写
state_dict = lrs.state_dict()
lrs.load_state_dict(state_dict)




#loss
##使用
loss_fn = Loss()
loss = loss_fn(out, target)
##类型
###reduction是reduce类型,可以选'none'|'elementwise_mean'|'sum'
loss_fn = nn.CrossEntropyLoss(weight=None, reduction='elementwise_mean') #weight是一维tensor,代表n类的权重
loss_fn = nn.MSELoss(reduction='elementwise_mean')
loss_fn = nn.L1Loss(reduction='elementwise_mean')
loss_fn = nn.NLLLoss(weight=None, reduction='elementwise_mean') #log_softmax + nll = cross_entropy
loss_fn = nn.NLLLoss2d(weight=None, reduction='elementwise_mean')
loss_fn = nn.KLDivLoss(weight=None, reduction='elementwise_mean') #没有维度限制的nll
loss_fn = nn.BCELoss(weight=None, reduction='elementwise_mean') #二分类,输出只有一个的交叉熵
loss_fn = nn.MarginRankingLoss(margin=0, reduction='elementwise_mean') #max(0,-y*(x1-x2)+margin),输入是(x1,x2,y)
loss_fn = nn.HingeEmbeddingLoss(margin=1, reduction='elementwise_mean') #xi,max(0,margin-xi)
loss_fn = nn.MultiLabelMarginLoss(reduction='elementwise_mean') #sum_ij max(0,1-(x[y[j]]-x[i]))
loss_fn = nn.SmoothL1Loss(reduction='elementwise_mean') #0.5(xi-yi)^2 if |xi-yi|<1 else |xi-yi|-0.5
loss_fn = nn.SoftMarginLoss(reduction='elementwise_mean') #二分类,log(1+exp(-y[i]*x[i]))
loss_fn = nn.MultiLabelSoftMarginLoss(weight=None, reduction='elementwise_mean') #y[i]log(e^x[i]/(1+e^[xi]))+(1-y[i])log(1/(1+e^x[i]))
loss_fn = nn.CosineEmbeddingLoss(margin=0, reduction='elementwise_mean') #{1-cos(x1,x2),max(0,cos(x1,x2)-margin)},输入是(x1,x2,y)
loss_fn = nn.MultiMarginLoss(p=1, margin=1, weight=None, reduction='elementwise_mean') #(max(0, margin-x[y]+x[i]))^p




#module
##层定义
###全连接层
fc1 = nn.Linear(in_features, out_features, bias=True)
embed = nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False) #?
###conv
'''
kernel_size,stride,padding,dilation可以是int或者tuple
dilation是卷积核元素间距
groups表示多少个并排互斥的卷积核通道组
'''
conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True)
###pool
'''
return_indices为true会返回最大值的下标
ceil_mode为true,在边界会采用上取整
count_include_pad为true,则计算会包括padding填充的0
'''
pool1 = nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
pool1 = nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
unpool1 = nn.MaxUnpool2d(kernel_size, stride=None, padding=0)
y = unpool1(x, indices, output_size=input.size())
pool1 = nn.FractionalMaxPool2d(kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None) #?
pool1 = nn.LPPool2d(norm_type, kernel_size, stride=None, ceil_mode=False) #?
pool1 = nn.AdaptiveMaxPool2d(output_size, return_indices=False) #global max pool
pool1 = nn.AdaptiveAvgPool2d(output_size) #global avg pool
###rnn
rnn1 = nn.RNN(input_size, hidden_size, num_layers, nonlinearity='tanh', bias=True, batch_first, dropout, bidirectional=False) #?
rnn1 = nn.LSTM(input_size, hidden_size, num_layers, bias=True, batch_first, dropout, bidirectional=False)
rnn1 = nn.GRU(input_size, hidden_size, num_layers, bias=True, batch_first, dropout, bidirectional=False)
rnn1 = nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh')
rnn1 = nn.LSTMCell(input_size, hidden_size, bias=True)
rnn1 = nn.GRUCell(input_size, hidden_size, bias=True)
###bn
bn1 = nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) #affine为false则去掉参数,track_running_stats为false则去掉buffer,永远算当前的均值和方差
bn1 = nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
###dropout
nn.Dropout(p=0.5) #drop_rate
nn.Dropout2d(p=0.5) #整个通道置0
###dist
dist = nn.PairwiseDistance(p=2, eps=1e-6, keepdim=False) #p=2就是欧氏距离
dist = nn.CosineSimilarity(dim=1, eps=1e-8)
###激活函数
nn.ReLU()
nn.ReLU6() #clip(x,0,6)
nn.ELU(alpha=1.0) #x,a(e^x-1)
nn.PReLU(num_parameters=1, init=0.25) #x,ax
nn.LeakyReLU(negative_slope=0.01) #x,ax
nn.Threshold(threshold, val) #x,val
nn.Hardtanh(min_value=-1, max_value=1) #clip(x,min,max)
nn.Sigmoid()
nn.Tanh()
nn.Softplus(beta=1, threshold=20) #1/beta*log(1+e^(beta*x))
nn.Softshrink(lambd=0.5) #x-l,0,x+l
nn.Softsign() #x/(1+|x|)
nn.Tanhshrink() #x-tanh(x)
nn.Softmin() #yi=e(-xi)/(sum_e(-xj))
nn.Softmax() #yi=e(xi)/(sum_e(xj))
nn.LogSoftmax() #log(softmax(x))
##detection
torchvision.ops.RoIPool(output_size, spatial_scale)
torchvision.ops.RoIAlign(output_size, spatial_scale, sampling_ratio)
##im2col
nn.Unfold(kernel_size, dilation=1, padding=0, stride=1) #im2col
nn.Fold(output_size, kernel_size, dilation=1, padding=0, stride=1) #col2im


##序列(为了把多个module合成一个调用)(参数是*args,或者是有序dict)
block = nn.Sequential(
    nn.Conv2d(1,6,3),
    nn.ReLU(True)
)
block = nn.Sequential(OrderedDict([
    ('conv1',nn.Conv2d(1,6,3)),
    ('relu1',nn.ReLU(True))
]))
y = block(x)
##列表(为了能分开调用,并且父亲model能遍历到它们)
blocks = nn.ModuleList([
    nn.Conv2d(1,6,3),
    nn.Conv2d(3,6,3)
])
y0 = blocks[0](x0)
y1 = blocks[1](x1)
##字典(理由同上)
blocks = ModuleDict({
    'conv0':nn.Conv2d(1,6,3),
    'conv1':nn.Conv2d(3,6,3),
})
y0 = blocks['conv0'](x0)
y1 = blocks['conv1'](x1)
##多parameter情况
params = nn.ParameterList([p1,p2,p3])
params = nn.ParameterDict({'a':p1,'b':p2})

##自定义module
class MyNet(nn.Module):
    def __init__(self):
        #添加module
        self.fc = nn.Linear(256,16)
        self.add_module("fc", nn.Linear(256,16))
        #添加临时变量(非参数)
        self.register_buffer(name, tensor)
        #添加参数
        self.register_parameter('w', Parameter(torch.Tensor(256,16)))
        self.w = Parameter(torch.Tensor(256,16))

    def forward(self, x):
        return x


##遍历
###state_dict
'''
一个有序字典,name->所有子孙module的parameter和buffer
'''
model.state_dict(destination=None, prefix='', keep_vars=False)#prefix是前缀
model.load_state_dict(state_dict)
model.load_state_dict(state_dict, strict=False) #有(漏了,多了)的情况不报错,而是返回一个列表
###参数生成器
model.parameters(recurse=True) #生成(p)
model.named_parameters(prefix='', recurse=True) #生成(name,p)
model.buffers(recurse=True)
model.named_buffers(prefix='', recurse=True)
###模块生成器
model.children() #儿子模块
model.named_children()
model.modules() #子孙模块
model.named_modules(prefix='')


##改变状态
###类型转换(和tensor一样)
model.to(torch.float64)
model.to(torch.device('cuda'))
model.cuda()
model.cpu()
###训练测试模式
model.training #是否是训练模式
model.train()
model.eval()
###梯度
model.zero_grad()
###万能
model.apply(fn)



#function
##全连接层
y = F.linear(x, w, bias=None)
##conv
y = F.conv2d(x, w, bias=None, stride=1, padding=0, dilation=1, groups=1)
y = F.conv_transpose2d(x, w, bias=None, stride=1, padding=0, output_padding=0, groups=1)
##pool
y = F.avg_pool2d(x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
y = F.max_pool2d(x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)
y = F.max_unpool2d(x, indices, kernel_size, stride=None, padding=0, output_size=None)
y = F.lp_pool2d(x, norm_type, kernel_size, stride=None, ceil_mode=False)
y = F.adaptive_max_pool2d(x, output_size, return_indices=False)
y = F.adaptive_avg_pool2d(x, output_size)
##bn
y = F.batch_norm(x, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05)
##dropout
y = F.dropout(x, p=0.5, training=False, inplace=False)
##dist
y = F.pairwise_distance(x1, x2, p=2, eps=1e-6)
y = F.cosine_similarity(x1, x2, dim=1, eps=1e-8)
##激活函数
###relu
y = F.relu(x, inplace=False)
y = F.relu6(x, inplace=False)
y = F.elu(x, alpha=1.0, inplace=False)
y = F.prelu(x, w)
y = F.leaky_relu(x, negative_slope=0.01, inplace=False)
y = F.rrelu(x, lower=0.125, upper=0.3333333333333333, training=False, inplace=False)
###softmax
y = F.softmax(x)
y = F.log_softmax(x)
y = F.sigmoid(x)
###
y = F.threshold(x, threshold, value, inplace=False)
y = F.hardtanh(x, min_val=-1.0, max_val=1.0, inplace=False)
y = F.logsigmoid(x)
y = F.hardshrink(x, lambd=0.5)
y = F.tanhshrink(x)
y = F.softsign(x)
y = F.softplus(x, beta=1, threshold=20)
y = F.softmin(x)
y = F.softshrink(x, lambd=0.5)
y = F.tanh(x)
##loss
y = F.nll_loss(x, target, weight=None, reduction='elementwise_mean')
y = F.kl_div(x, target, reduction='elementwise_mean')
y = F.binary_cross_entropy(x, target, weight=None, reduction='elementwise_mean')
y = F.smooth_l1_loss(x, target, reduction='elementwise_mean')
##图像
y = F.pad(x, pad, mode='constant', value=0)
##正则
y = F.normalize(x, p=2, dim=1, eps=1e-12) #除以p范数
##detection
torchvision.ops.nms(boxes, scores, iou_threshold)
torchvision.ops.roi_pool(input, boxes, output_size, spatial_scale=1.0)
torchvision.ops.roi_align(input, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1)
##im2col
x_col = F.Unfold(x, kernel_size, dilation=1, padding=0, stride=1) #im2col
x = F.Fold(x_col, output_size, kernel_size, dilation=1, padding=0, stride=1) #col2im
'''
#实现卷积
x = torch.randn(64,3,28,28)
w = torch.randn(20,3,5,5)
x_col = F.unfold(x,(5,5))
y_col = x_col.transpose(1,2).matmul(w.view(w.shape[0],-1).t()).transpose(1,2)
y = F.fold(y_col, (24,24), (1,1))
'''


##自定义函数
'''
forward内部所有变量都没有梯度,因为Function就是计算图的最小单元
'''
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<0] = 0
        return grad_input

class XX(torch.autograd.Function):
    def __init__(self):
        pass

    def forward(self, x1, x2):
        return y1,y2

    def backward(self, grad_y1, grad_y2):
        return grad_x1,grad_x2
        #return grad_x1,None

##hook

###nn.Module
def forward_hook_fn(module, inputs, outputs):
    '''
    inputs: list
    outputs: tensor/list
    '''
    #不能修改输入和输出
    #一定要返回None
    return None

def backward_hook_fn(module, grad_inputs, grad_outputs):
    '''
    grad_inputs:list
    grad_outputs:list
    '''
    #不能修改输入和输出
    #但你可以返回另一个张量作为输入的梯度
    #backward只能作用在model的最后一个Function
    return another_grad_inputs

def forward_per_hook_fn(module, inputs):
    '''
    inputs: list
    '''
    #不能修改输入
    #一定要返回None
    return None

handle = model.register_forward_hook(forward_hook_fn)
handle = model.register_backward_hook(backward_hook_fn)
handle = model.register_forward_pre_hook(forward_per_hook_fn)
handle.remove() #去掉这个hook

###Tensor
def hook_fn(grad_x):
    #它相当于backward_hook
    #用来修改某个张量的梯度
    return modified_grad_x

handle = x.register_hook(hook_fn)


#transform
##组合
mytransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
])

##针对PIL图像
###裁剪
t = transforms.CenterCrop(size) #size是int或者(int,int)
t = transforms.RandomCrop(size, padding=0)
t = transforms.RandomSizedCrop(size, interpolation=2) #随机切+resize
###扩充
t = transforms.Pad(padding, fill=0) #padding是int或者(int,int),或者(int)*4
###resize
t = transforms.Resize(size, interpolation=2) #interpolation是resize的方法
###翻转
t = transforms.RandomHorizontalFlip()
t = transforms.RandomVerticalFlip()
###光照
t = transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0) #随机改变亮度,对比度,饱和度,色调
###颜色空间
t = transforms.Grayscale(num_output_channels=1) #变成灰度图,参数是1或3(r=g=b)
t = transforms.RandomGrayscale(p=0.1) #变成灰度图时随机扰动

##针对tensor
###变换
t = transforms.Normalize(mean, std) #mean和std是一维数组

##转换
t = transforms.ToTensor() #[0,255]->[0,1],[h,w,c]->[c,h,w]
t = transforms.ToPILImage()

##tensor和PIL图像通用
t = transforms.Lambda(lambd)


#models
#pretrained是都有的参数
model = models.alexnet(pretrained=False)
model = models.vgg16() #11,13,16,19
model = models.vgg16_bn() #11,13,16,19
model = models.googlenet(aux_logits=False)
model = models.inception_v3(aux_logits=False)
model = models.resnet18() #18,34,50,101,152
model = models.resnext50_32x4d() #(50,32,4),(101,32,8)
model = models.wide_resnet50_2() #(50,101)
model = models.densenet_161(memory_efficient=False) #121,161,169,201

model = models.squeezenet1_0() #1.0,1.1
model = models.mobilenet_v2()
model = models.shufflenet_v2_x0_5() #0.5,1.0,1.5,2.0
model = models.mnasnet0_5() #0.5,0.75,1.0,1.3

model = models.detection.fasterrcnn_resnet50_fpn(num_classes=91, pretrained_backbone=True)
model = models.detection.maskrcnn_resnet50_fpn(num_classes=91, pretrained_backbone=True)
model = models.detection.keypointrcnn_resnet50_fpn(num_classes=91, pretrained_backbone=True)

model = models.segmentation.fcn_resnet50(num_classes=21) #50,101
model = models.segmentation.deeplabv3_resnet50(num_classes=21) #50,101


#dataset

##自定义
class MyDataset(data.Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

##数据集拼接
data.ConcatDataset([d1,d2,d3]) 
##数据集子集
data.Subset(dataset, indices) #indices是保留的ix
##k折划分
d1,d2,d3 = data.random_split(dataset, 3)

##来自tensor
data.TensorDataset(x, y, z) #tensor的第0维是样本维度
##来自文件夹
###图片文件夹
datasets.ImageFolder(root, transform=None, target_transform=None)
'''
root文件夹结构:
root/dog/xxx.png
root/dog/xxy.png

root/cat/123.png
root/cat/nsdf3.png
'''
###一般文件夹
datasets.DatasetFolder(root, loader, extensions=None, transform=None, target_transform=None)
'''
loader是一个输入路径读入数据并返回的函数
extensions是一个存放合法文件扩展名的列表,如['jpg','png']
'''

##网上数据集
##transform,target_transform,download是都有的参数
datasets.MNIST(root, train=True, transform=None, target_transform=None, download=False)
datasets.CIFAR10(root, train=True)
datasets.CIFAR100(root, train=True)

datasets.STL10(root, split='train')
datasets.SVHN(root, split='train')
datasets.ImageNet(root, split='train')

datasets.CocoDetection(root, annFile)
datasets.CocoCaptions(root, annFile)
datasets.VOCDetection(root, year='2012', image_set='train')
datasets.VOCSegmentation(root, year='2012', image_set='train')


#dataloader
'''
sampler:提取样本的策略,如果有则忽略shuffle参数
collate_fn:样本聚合成batch的函数
pin_memory:是否自动将数据拷贝到cuda上
drop_last:如果batch不整除,则删掉最后一个
'''
loader = data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False)
train_loader = data.DataLoader(mnist_train, batch_size=16, shuffle=False, num_workers=2)
for i,data in enumerate(train_loader):
    print(data)

#sampler
sampler = data.SequentialSampler(dataset) #顺序
sampler = data.RandomSampler(dataset) #随机




#init
init.constant_(w, val)
init.uniform_(w, a=0, b=1) #均匀分布
init.normal_(w, mean=0, std=1)
init.eye_(w) #单位矩阵
init.dirac_(w) #?
init.xavier_uniform_(w, gain=1) #glorot,gain是倍数,bound=sqrt(6/(i+o))
init.xavier_normal_(w, gain=1) #std=sqrt(2/(i+o))
init.kaiming_uniform_(w, a=0, mode='fan_in') #mode还可以是fan_out,bound=sqrt(6/((1+a2)*mode))
init.kaiming_normal_(w, a=0, mode='fan_in') #std=sqrt(2/((1+a2)*mode))
init.orthogonal_(w, gain=1) #?
init.sparse_(w, sparsity, std=0.01) #?



#gpu and cuda
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,3'

export CUDA_HOME="/usr/lib/local/cuda-9.1"
export LD_LIBRARY_PATH="/usr/lib/local/cuda-9.1/lib64"

torch.version.cuda


#图片格子可视化
'''
tensor是4d tensor(B,3,H,W)或一个list的3d tensor
padding控制图片边距

返回值是3d tensor(3,H,W)
'''
tensor = torchvision.utils.make_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
#处理并把图片保存下来
torchvision.utils.save_image(tensor, filename, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)


'''
1.bn的momentum和tf的是反过来的
2.optim的weight_decay会把所有参数(包括bias)都加上正则项
3.adam之所有不好,是因为它有平方,所以精度容易丢失
'''