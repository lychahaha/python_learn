import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.backends import cudnn
import torch.utils.data as data
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

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


#全局改变
##改变形状
y = torch.squeeze(x, dim=None) #(共享内存)
y = torch.unsqueeze(x, dim) #expand_dim
x = torch.cat(inputs, dimension=0) #concat
ys = torch.chunk(x, chunk_size, dim=0) #?
##改变迭代方向
y = torch.t(x) #0,1维转置
y = torch.transpose(x, dim0, dim1)
##连接与分割
y = torch.stack(inputs, dim=0)
ys = torch.unbind(x, dim=0) #unstack
ys = torch.split(x, split_size, dim=0) #?


#下标
y = torch.gather(x, dim, index) #index和gather的shape一样
y = torch.index_select(x, dim, index) #index是一维数组
y = torch.masked_select(x, mask) #?
y = torch.nonzero(x) #返回的shape是(none,x_dims)


#生成
##随机
x = torch.rand(5, 3) #01均匀分布
x = torch.randn(5, 3) #标准正态分布
x = torch.randperm(n)
##填充
x = torch.ones(5, 3)
x = torch.zeros(5, 3)
##数列
x = torch.arange(start, end, step=1)
x = torch.linspace(start, end, steps=100) #左闭右闭
x = torch.logspace(start, end, steps=100)
#矩阵
x = torch.eye(n, m=None)
y = torch.diag(x, diagonal=0) #矩阵<->对角线,diagonal控制对角线(大于0向上偏移)




y = torch.bernoulli(x) #x<0.5->y=0,x>=0.5->y=1(x的值域必须是[0,1])
y = torch.multinomial(x, num_samples, replacement=False) #?
x = torch.normal(means, stds) #?


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
##?
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
##大小于
z = torch.eq(x, y)
z = torch.ge(x, y)
z = torch.gt(x, y)
z = torch.le(x, y)
z = torch.lt(x, y)
z = torch.ne(x, y)
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


#tensor
##构造
x = torch.Tensor(5, 3)
x = torch.FloatTensor([0.1,0.2,0.3])
##属性
a = x.size() #shape
a = torch.numel(x) #size
a = x.dim() #维数
a = x.data_ptr() #头指针
a = x.element_size() #数据类型的字节数
##特殊方法
x.fill_(1.2)
x.map_(y, fx) #def fx:(a,b)->c
x.repeat(*sizes) #?
x.resize_(*sizes) #?
x.scatter(y, dim, index) #?
x.select(dim, index) #?
##reshape
y = x.view(3, 5)
y = x.view(-1, 5)
##复制
y = x.clone()
y = x.contiguous() #如果内存连续,则返回x本身
##转换
y = x.type(torch.FloatTensor)
##类型
torch.FloatTensor #默认
torch.DoubleTensor
torch.cuda.HalfTensor #16位浮点数
torch.ByteTensor #unsigned
torch.CharTensor
torch.ShortTensor
torch.IntTensor
torch.LongTensor


#variable
##构造
x = Variable(torch.ones(2,2), requires_grad=True) #requires_grad表示backward时是否需要计算其梯度
x = Variable(torch.ones(2,2), volatile=True) #适用于inference时输入设置,自己和后续不会保存中间状态
##成员
y = x.grad #variable
y = x.data #tensor
op = x.creator #f


a = torch.cuda.device_count()


a = torch.get_num_threads()
torch.set_num_threads(a)


#转换
x.numpy()
torch.from_numpy(a) #共享内存
x.cuda()


torch.cuda.is_available()


torch.cuda.device(1)



x.backward(k)
x.backward()


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
##不同参数不同超参数
opt = optim.SGD(
    [
        {'params':model.base.parameters()},
        {'params':model.classifier.parameters(), 'lr':1e-3}
    ]
    lr=1e-2, momentum=0.9
)
##方法
optimizer.zero_grad()
optimizer.step()
optimizer.load_state_dict(state_dict)
d = optimizer.state_dict()


#loss
##使用
loss_fn = Loss()
loss = loss_fn(out, target)
##类型
loss_fn = nn.CrossEntropyLoss(weight=None, size_average=True) #weight是一维tensor,代表n类的权重
loss_fn = nn.MSELoss(size_average=True)
loss_fn = nn.L1Loss(size_average=True)
loss_fn = nn.NLLLoss(weight=None, size_average=True) #log_softmax + nll = cross_entropy
loss_fn = nn.NLLLoss2d(weight=None, size_average=True)
loss_fn = nn.KLDivLoss(weight=None, size_average=True) #没有维度限制的nll
loss_fn = nn.BCELoss(weight=None, size_average=True) #二分类,输出只有一个的交叉熵
loss_fn = nn.MarginRankingLoss(margin=0, size_average=True) #max(0,-y*(x1-x2)+margin),输入是(x1,x2,y)
loss_fn = nn.HingeEmbeddingLoss(margin=1, size_average=True) #xi,max(0,margin-xi)
loss_fn = nn.MultiLabelMarginLoss(size_average=True) #sum_ij max(0,1-(x[y[j]]-x[i]))
loss_fn = nn.SmoothL1Loss(size_average=True) #0.5(xi-yi)^2 if |xi-yi|<1 else |xi-yi|-0.5
loss_fn = nn.SoftMarginLoss(size_average=True) #二分类,log(1+exp(-y[i]*x[i]))
loss_fn = nn.MultiLabelSoftMarginLoss(weight=None, size_average=True) #y[i]log(e^x[i]/(1+e^[xi]))+(1-y[i])log(1/(1+e^x[i]))
loss_fn = nn.CosineEmbeddingLoss(margin=0, size_average=True) #{1-cos(x1,x2),max(0,cos(x1,x2)-margin)},输入是(x1,x2,y)
loss_fn = nn.MultiMarginLoss(p=1, margin=1, weight=None, size_average=True) #(max(0, margin-x[y]+x[i]))^p


model = nn.DataParallel(model)


param.requires_grad = False #不更新权重


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
pool1 = nn.AdaptiveMaxPool2d(output_size, return_indices=False) #?
pool1 = nn.AdaptiveAvgPool2d(output_size) #?
###rnn
rnn1 = nn.RNN(input_size, hidden_size, num_layers, nonlinearity='tanh', bias=True, batch_first, dropout, bidirectional=False) #?
rnn1 = nn.LSTM(input_size, hidden_size, num_layers, bias=True, batch_first, dropout, bidirectional=False)
rnn1 = nn.GRU(input_size, hidden_size, num_layers, bias=True, batch_first, dropout, bidirectional=False)
rnn1 = nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh')
rnn1 = nn.LSTMCell(input_size, hidden_size, bias=True)
rnn1 = nn.GRUCell(input_size, hidden_size, bias=True)
###bn
bn1 = nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True)
###dropout
nn.Dropout(p=0.5) #drop_rate
nn.Dropout2d(p=0.5) #整个通道置0
###p范数
dist = nn.PairwiseDistance(p=2, eps=1e-6)
y = dist(x)
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
##序列
conv = nn.Sequential(
    nn.Conv2d(1,6,3),
    nn.ReLU(True)
)

params = model.parameters()
model.zero_grad()

class MyNet(nn.Module):
    def forward(self, x):
        return x

    def print_fn(self, inputs, output):
        pass

    def print_fn(self, grad_input, grad_output):
        pass

conv1.register_forward_hook(print_fn)
conv1.register_backward_hook(print_fn)

model.train

model.cuda()

model.train()
model.eval()


#function
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
##激活函数
###relu
y = F.relu(x, inplace=False)
y = F.relu6(x, inplace=False)
y = F.prelu(x, w)
y = F.rrelu(x, lower=0.125, upper=0.3333333333333333, training=False, inplace=False)
y = F.elu(x, alpha=1.0, inplace=False)
y = F.leaky_relu(x, negative_slope=0.01, inplace=False)
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
##bn
y = F.batch_norm(x, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05)
##全连接层
y = F.linear(x, w, bias=None)
##dropout
y = F.dropout(x, p=0.5, training=False, inplace=False)
##dist
y = F.pairwise_distance(x1, x2, p=2, eps=1e-6)
##loss
y = F.nll_loss(x, target, weight=None, size_average=True)
y = F.kl_div(x, target, size_average=True)
y = F.binary_cross_entropy(x, target, weight=None, size_average=True)
y = F.smooth_l1_loss(x, target, size_average=True)
##图像
y = F.pad(x, pad, mode='constant', value=0)

##自定义函数
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, =ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<0] = 0
        return grad_input


#transform
##组合
mytransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
])
##裁剪
t = transforms.CenterCrop(size) #size是int或者(int,int)
t = transforms.RandomCrop(size, padding=0)
t = transforms.RandomSizedCrop(size, interpolation=2) #随机切+resize
##扩充
t = transforms.Pad(padding, fill=0) #padding是int或者(int,int),或者(int)*4
##resize
t = transforms.Resize(size, interpolation=2) #interpolation是resize的方法
##翻转
t = transforms.RandomHorizontalFlip()
t = transforms.RandomVerticalFlip()
##转换
t = transforms.ToTensor() #[0,255]->[0,1],[h,w,c]->[c,h,w]
t = transforms.ToPILImage()
##变换
t = transforms.Normalize(mean, std) #mean和std是一维数组
##光照
t = transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0) #随机改变亮度,对比度,饱和度,色调
##颜色空间
t = transforms.Grayscale(num_output_channels=1) #变成灰度图,参数是1或3(r=g=b)
t = transforms.RandomGrayscale(p=0.1) #变成灰度图时随机扰动
##通用
t = transforms.Lambda(lambd)


#models
alexnet = models.alexnet(pretrained=False)
vggnet = models.vgg16() #11,13,16,19
inception = models.inception_v3()
resnet18 = models.resnet18() #18,34,50,101,152
squeezenet = models.squeezenet1_0() #1.0,1.1
densenet = models.densenet_161() #121,161,169,201


#保存读取
torch.save(model, 'model.pkl') #保存整个神经网络的结构和模型参数
model = torch.load('model.pkl')
torch.save(model.state_dict(), 'params.pkl') #只保存神经网络的模型参数
model_object.load_state_dict(torch.load('params.pkl'))


#dataset
##网上数据集
mnist_train = datasets.MNIST(root, train=True, transform=None, target_transform=None, download=False)
datasets.CocoCaptions(root="dir where images are", annFile="json annotation file", [transform, target_transform])
datasets.CocoDetection(root="dir where images are", annFile="json annotation file", [transform, target_transform])
datasets.LSUN(db_path, classes='train', [transform, target_transform])
datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=False)
datasets.CIFAR100(root, train=True, transform=None, target_transform=None, download=False)
datasets.STL10(root, split='train', transform=None, target_transform=None, download=False)
##来自文件夹
datasets.ImageFolder(root="root folder path", [transform, target_transform])
##自定义
class MyDataset(data.Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


#dataloader
'''
sampler:提取样本的策略,如果有则忽略shuffle参数
collate_fn:?
pin_memory:是否自动将数据拷贝到cuda上
drop_last:如果batch不整除,则删掉最后一个
'''
loader = data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False)
train_loader = data.DataLoader(mnist_train, batch_size=16, shuffle=False, num_workers=2)
for i,data in enumerate(train_loader):
    print(data)


#init
init.constant(w, val)
init.uniform(w, a=0, b=1) #均匀分布
init.normal(w, mean=0, std=1)
init.eye(w) #单位矩阵
init.dirac(w) #?
init.xavier_uniform(w, gain=1) #glorot,gain是倍数
init.xavier_normal(w, gain=1)
init.kaiming_uniform(w, a=0, mode='fan_in')
init.kaiming_normal(w, a=0, mode='fan_in')
init.orthogonal(w, gain=1) #?
init.sparse(w, sparsity, std=0.01) #?
