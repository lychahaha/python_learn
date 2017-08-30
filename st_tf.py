import tensorflow as tf
import numpy as np

#会话
sess = tf.Session()
sess = tf.InteractiveSession()#支持eval
sess.run(res)
sess.run(res, feed_dict={input1:val1, input2:val2})
sess.close()

#常量
a = tf.constant(10)
b = tf.constant('Hello')
c = tf.constant([3.,3.])
d = tf.constant(2, name='d')
e = tf.constant(3, shape=[4,5])#相当于fill
##
a = tf.zeros([200])
a = tf.zeros_like(a)#创建形状和张量a一样的全0张量
a = tf.ones([3,4])
a = tf.ones_like(a)
a = tf.fill([2,3], 4)
a = tf.linspace(first,last,num)#创建线性增长,长度为num的一维张量
a = tf.range(beg,end,dx)
##随机
a = tf.random_uniform([32,32], minval=0.0, maxval=1.0)
a = tf.random_normal([32,32], mean=-1, stddev=0.35)
a = tf.truncated_normal([32,32])#截尾的正态分布(截去2stddev以上)
c = tf.random_shuffle(a)#只随机打乱最高维
###
tf.set_random_seed(seed)#设置随机种子

#变量
a = tf.Variable([1.,2.])
a = tf.Variable(3, name='a')
a = tf.get_variable('a', shape=[2,3], initializer=tf.constant_initializer(1))
##属性方法
a.name
a.dtype
a.get_shape()
a.device
a.initialized_value()#其他变量的初始值

#输入变量
a = tf.placeholder(tf.float32)
a = tf.placeholder(tf.float32, [2,3])
a = tf.placeholder(tf.float32, shape=(2,3))
a = tf.placeholder(tf.float32, [None,3], name='input')





#op
##基本运算符
c = tf.add(a,b)#subtract,multiply,div,mod
##数学函数
c = tf.log(a)
c = tf.exp(a)
c = tf.pow(a,3)
c = tf.sqrt(a)
c = tf.square(a)#y=x*x
c = tf.rsqrt(a)#y=1/sqrt(x)
c = tf.neg(a)#y=-x
c = tf.inv(a)#y=1/x
c = tf.sin(a)#cos,tan,atan...
##非数学类函数
c = tf.abs(a)
c = tf.sign(a)
c = tf.ceil(a)
c = tf.floor(a)
c = tf.round(a)
c = tf.maximum(a,b)
c = tf.minimum(a,b)
##矩阵运算
c = tf.matmul(a,b)#矩阵乘法
c = tf.matmul(a, b, transpose_a=False, transpose_b=False, a_is_sparse=False, b_is_sparse=False, name=None)
c = tf.diag(a)#根据主元值的张量(向量),生成对角张量(矩阵)
c = tf.transpose(a)#转置
c = tf.transpose(a, perm=[0,2,1])#张量转置(维度置换)
c = tf.matrix_determinant(a)#行列式
c = tf.matrix_inverse(a)#逆
c = tf.cholesky(a)#矩阵cholesky分解:A=C*(CT)
c = tf.trace(a)#矩阵的迹
##复数运算
c = tf.complex(a,b)#返回a+bi
c = tf.complex_abs(a)#返回复数模长
c = tf.conj(a)#返回复数的共轭
c = tf.real(a)#返回复数的实部
c = tf.imag(a)#返回复数的虚部
##规约运算
c = tf.reduce_sum(a)
c = tf.reduce_sum(a, reduction_indices=None, keep_dims=False, name=None)#reduction_indices表明按哪个轴规约
c = tf.reduce_mean(a)
c = tf.reduce_prod(a)#求积
c = tf.reduce_min(a)
c = tf.reduce_max(a)
###特殊规约
c = tf.accumulate_n([a,b,...])#n个tensor相加
###布尔规约运算
c = tf.reduce_all(a)
c = tf.reduce_any(a)
##分段规约运算(按最高维度分段,相同id进行规约)
c = tf.segment_sum(a,seg_ids)#seg_id是向量,需要升序
c = tf.segment_prod(a,seg_ids)
c = tf.segment_min(a,seg_ids)
c = tf.segment_max(a,seg_ids)
c = tf.segment_mean(a,seg_ids)
c = tf.segment_(a,seg_ids)
c = tf.segment_prod(a,seg_ids)
c = tf.unsorted_segment_sum(a,seg_ids)#seg_id不需要升序
##下标运算
c = tf.argmax(a, dimension)#dimension表明按照哪个维度进行规约
c = tf.argmin(a, dimension)
c = tf.listdiff(a, ids)#根据下标ids在a里取值
c = tf.invert_permutation(a)#求置换a的逆,a必须是个排列
###布尔下标运算
c = tf.where(a)#返回a中每个true的坐标,输出是2维(每个true,坐标的每个维度)
###去重
c = tf.unique(a)#去重,输入必须是1维张量
##逻辑运算
c = tf.logical_and(a,b)
c = tf.logical_or(a,b)
c = tf.logical_not(a)
c = tf.logical_xor(a,b)
##布尔运算
c = tf.equal(a,b)
c = tf.not_equal(a,b)
c = tf.less(a,b)
c = tf.less_equal(a,b)
c = tf.greater(a,b)
c = tf.greater_equal(a,b)
c = tf.select(condition,t,e)#c,t,e和输出形状相同,c是布尔类型,为true选择t对应的元素,否则选择e
##激活函数
c = tf.nn.relu(a)#max(a,0)
c = tf.nn.relu6(a)#min(relu(a),6)
c = tf.nn.softplus(a)#log(exp(a)+1)
c = tf.sigmoid(a)#1/(1+exp(-a))
c = tf.tanh(a)
##类型转换
c = tf.cast(a, tf.float32)
c = tf.string_to_number(a)
c = tf.to_double(a)
c = tf.to_float(a)
c = tf.to_bfloat16(a)
c = tf.to_int32(a)
c = tf.to_int64(a)
c = tf.to_float(a)
c = tf.to_float(a)
c = tf.to_float(a)
c = tf.to_float(a)
##其他操作
c = tf.reshape(a, shape=[2,3,4])
c = tf.squeeze(a, squeeze_dims=None)#去掉长度为1的维度,如果squeeze_dims是一个列表,则去掉列表中指定的维度
c = tf.expand_dims(a, dim=3)#在第dim个维度前插入长度为1的维度
c = tf.concat([a,b], 1)#1表示连接的轴
##属性
c = tf.shape(a)
c = tf.size(a)
c = tf.rank(a)


#update = tf.assign(c,a)
##赋值
update = a.assign(b)
update = a.assign_add(b)
update = a.assign_sub(b)


#初始化
tf.constant_initializer(1)
tf.contrib.layers.xavier_initializer()
tf.contrib.layers.variance_scaling_initializer()#he init



#变量初始化
init_op = tf.global_variables_initializer()



#指定CPU,GPU
tf.device("/cpu:0")
tf.device("/gpu:0")


#保存与载入
saver = tf.train.Saver()
save_path = saver.save(sess, "/tmp/model.ckpt")
saver.restore(sess, "/tmp/model.ckpt")

#数据类型
tf.float16,tf.float32,tf.float64,
tf.bfloat16,tf.complex64,tf.complex128,
tf.int8,tf.uint8,tf.uint16,tf.int16,tf.int32,
tf.int64,tf.bool,tf.string



######################

##batch norm
from tensorflow.contrib.layers.python.layers import batch_norm

def batch_norm_layer(inputT, is_training, scope):
	return tf.cond(is_training,
		lambda: batch_norm(inputT, is_training=True,
		center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9, scope=scope),
		lambda: batch_norm(inputT, is_training=False,
		center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9,
		scope=scope, reuse = True))

#加上滑动均值和方差的更新
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
train_ops = [train_step] + update_ops
train_op_final = tf.group(*train_ops)

#手动
mean, variance = tf.nn.moments(x, axis)
update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

mean, variance = tf.cond(is_train, lambda: (mean, variance), lambda: (moving_mean, moving_variance))

x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)




##求batch的均值和方差
#axises一般为除最后一维的所有维度
batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')



##变量作用域
with tf.variable_scope('V1'):  
	a1 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))  
#重用
with tf.variable_scope('V1', reuse=True):  
	a3 = tf.get_variable('a1')

#?
with tf.name_scope('xxx'):
	a = tf.Variable([1,3,4])



##条件
#condition是一个tf.bool的张量
#f1,f2为py函数
#true->f1,false->f2
ret = tf.cond(condition,f1,f2)



##控制依赖
#在运算之前进行更新
a = tf.Variable(1.0)
b = tf.Variable(2.0)
update = tf.assign(b,b+1)

with tf.control_dependencies([update]):
	out = a*b
out2 = a*b
#输出out=3.0,out2=2.0



##?
tf.identity(a)



##变量集合
#存
tf.add_to_collection('loss', v1)
#取
l = tf.get_collection('loss')



##log??
tf.summary.scalar('loss', V1)
s_op = tf.summary.merge_all()



##梯度提取和处理
#优化器
opt = GradientDescentOptimizer(learning_rate=0.1)
# grads_and_vars为tuples (gradient, variable)组成的列表
grads_and_vars = opt.compute_gradients(loss, <list of variables>)
#梯度稍作处理
capped_grads_and_vars = [(MyCapper(gv[0]), gv[1]) for gv in grads_and_vars]
#应用梯度
opt.apply_gradients(capped_grads_and_vars)



##衰退的学习率
#decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
#以0.96为基数，每100000 步进行一次学习率的衰退
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,100000, 0.96, staircase=True)
learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)



##flag(常量)
flags = tf.app.flags
FLAGS = flags.FLAGS
#定义(名字,值,描述)
flags.DEFINE_bool('is_ok', False, 'help desc')
flags.DEFINE_integer('epochs', 10, 'train times')
flags.DEFINE_string('data_dir', '/tmp/data', 'dir of data')
#使用
print(FLAGS.data_dir)



#tensor board
tensorboard --logdir=/tmp/mnist_logs/train/ -port=2213



#save and restore
saver = tf.train.Saver()

#save
saver.save(sess, r'D:\newcode\model.ckpt')#一定要全路径
#restore
saver.restore(sess, r'D:\newcode\model.ckpt')



