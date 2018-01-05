#jh:mix-base,lazy-type
import tensorflow as tf
import numpy as np


#keras.layers
from tensorflow.contrib.keras.python.keras.layers import Conv2D,Dense,MaxPool2D,ZeroPadding2D,Flatten,BatchNormalization,Dropout,AvgPool2D,Activation
from tensorflow.contrib.keras.python.keras.layers import Conv3D,MaxPool3D,AvgPool3D,ZeroPadding3D
from tensorflow.contrib.keras.python.keras import layers
from tensorflow.contrib.keras import backend as K
##conv
##变量放在TRAINABLE_VARIABLES
x = Conv2D(64, (3,3), strides=(1,1), padding='valid', name='conv1')(x) #w:'conv1/kernel', b:'conv1/bias'
x = Conv2D(32, (3,3), padding='same', activation='relu')(x)
##fc
##变量放在TRAINABLE_VARIABLES
x = Dense(256, activation='relu', name='fc1')(x) #w:'fc1/kernel', b:'fc1/bias'
##pool
x = MaxPool2D((2,2), strides=(2,2))(x)
x = AvgPool2D((2,2), strides=(2,2))(x)
##batch normalization
##beta和gamma放在TRAINABLE_VARIABLES
##更新操作放在UPDATE_OPS
x = BatchNormalization(axis=3, name='bn1')(x) #'bn1/beta', 'bn1/gamma', 'bn1/moving_mean', 'bn1/moving_variance',
##activation
x = Activation('relu')(x)
##flatten
x = Flatten()(x)
##dropout
x = Dropout(drop_rate)(x) #注意是drop的比例
feed[K.learning_phase()] = int(is_train) #需要加上这
##补零
x = ZeroPadding2D((3,3))(x) #两边都加3
##3D
x = Conv3D(32, (3,3,3), strides=(2,2,2), padding='SAME', name='conv1')(x)
x = MaxPool3D((2,2,2), strides=(2,2,2))(x)
x = AvgPool3D((2,2,2), strides=(2,2,2))(x)
x = ZeroPadding3D((3,3,3))(x)

#tf.layers
##fc
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
x = tf.matmul(x,w) + b
##conv
w = tf.Variable(tf.zeros([3,3,1,16]))
x = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')
##pool
x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
##dropout
x = tf.nn.dropout(x, keep_rate)
##flatten
x = tf.reshape(x, [-1,np.prod(x.get_shape()[1:]).value])
##batch normalization
shape = [x.get_shape()[3].value]
beta = tf.get_variable('beta', shape=shape)
gamma = tf.get_variable('gamma', shape=shape)
mean,var = tf.nn.moments(x, [0,1,2]) #求均值和方差,axes表示对哪些轴取统计值
eme = tf.train.ExponentialMovingAverage(decay)
maintain_op = eme.apply([mean,var])
tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, maintain_op)
moving_mean = eme.average(mean)
moving_var = eme.average(var)
cur_mean,cur_var = tf.cond(is_train, lambda: (mean,var), lambda: (moving_mean,moving_var))
x = tf.nn.batch_normalization(x, mean=cur_mean, variance=cur_var, offset=beta, scale=gamma, variance_epsilon=1e-8)
##deconv
w = tf.Variable(tf.zeros([3,3,1,16]))
x = tf.nn.conv2d_transpose(x, w, output_shape=[None,224,224,1], strides=[1,1,1,1], padding='SAME')


#sess
sess = tf.Session()
##带config
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True #显存按需增长
sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4 #设置占用比例
sess_config.allow_soft_placement = True #设备不合适时,自动寻找合适的设备
sess_config.log_device_placement = True
sess_config.operation_timeout_in_ms = 5000 #设置timeout自动报错,查看卡死的地方
sess = tf.Session(config=sess_config)
##可以支持eval的
sess = tf.InteractiveSession()
##运行
ans = sess.run(obj, feed_dict={a1:b1,a2:b2})
##关闭
sess.close()


#tensor<->list(tensor)
##list->tensor
t = tf.stack(l, axis=0) #创建新轴
c = tf.concat([a,b,c], 2) #不创建新轴
##tensor->list
l = tf.unstack(t, axis=0)
l = tf.split(t, num_split, axis=0) #num_split必须整除axis的长度
##n个张量相加
t = tf.add_n(l)


#改变形状
##reshape
c = tf.reshape(a, [2,3,4])
##去掉长度为1的维度
c = tf.squeeze(a) #去掉所有
c = tf.squeeze(a, axis=[1,4]) #去掉指定的
##插入长度为1的维度
c = tf.expand_dims(a, axis=2)
##转置
c = tf.transpose(a)
c = tf.transpose(a, perm=[3,0,1,2])
##翻转
c = tf.reverse(a, [0,1]) #后面参数是要翻转的轴
#改变大小
##平铺
##每个维度一个数,表示重复次数
##a.shape=[3,4,5],c.shape=[6,12,20](shape对应相乘)
c = tf.tile(a, [2,3,4])
##扩张
##每个维度两个数,表示上下的扩张数
##a.shape=[2,3,4],c.shape=[2+1+2,3+3+4,4+5+6]=[5,10,15]
c = tf.pad(a, [[1,2],[3,4],[5,6]], constant_values=0)


#复杂函数
##激活函数
x = tf.nn.softmax(x)
x = tf.nn.relu(x)
x = tf.nn.crelu(x) #[relu(x),relu(-x)],最后的维度会加倍
x = tf.nn.relu6(x) #clip(x, 0, 6)
x = tf.nn.tanh(x)
x = tf.nn.sigmoid(x)
x = tf.nn.elu(x) #e^x-1或x,分界线是x=0
x = tf.nn.softplus(x) #log(exp(x)+1)
x = tf.nn.softsign(x) #x/(|x|+1)
x = tf.nn.bias_add(x, bias) #x+bias,没啥用
##loss
loss = tf.nn.l2_loss(x)
x = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels) #先softmax,再求交叉熵,返回一个向量(每条记录产生的交叉熵)
x = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) #label是没有onehot的
x = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels) #softmax换成sigmoid
##top-k
x = tf.nn.in_top_k(logits, labels, k) #logits是二维的,labels是一维的
##onehot
c = tf.one_hot(a, cla_nums)
##clip
c = tf.clip_by_value(a, minval, maxval)
c = tf.clip_by_norm(a, 1.0, axes=[2]) #限制第2条轴规约的L2范数小于等于1.0
##正则化
c = tf.norm(a, ord=2, axis=None, keep_dims=False) #求p范数


#收集
##add
tf.add_to_collection('loss', a)
##get
l = tf.get_collection('loss')
l = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='conv')
##tf.GraphKeys
tf.GraphKeys.VARIABLES #tf的变量会自动放到这里
tf.GraphKeys.TRAINABLE_VARIABLES #tf的可训练变量会自动放到这里
tf.GraphKeys.UPDATE_OPS #keras的bn会将滑动平均更新op放在这里
tf.GraphKeys.SUMMARIES #tf的summary会自动放到这里


#保存与载入
##saver
saver = tf.train.Saver() #默认所有变量
saver = tf.train.Saver([a,b,c]) #指定要保存(载入)的变量
saver = tf.train.Saver({'a':a,'b':b}) #指定(文件数据名字->载入变量),这样即使变量名变了也没关系
##save
save_path = saver.save(sess, '/tmp/model.ckpt')
save_path = saver.save(sess, '/tmp/model.ckpt', global_step=3)
##load
##restore是根据saver初始化的变量列表,去文件寻找对应的数据
##因此变量列表<=文件数据列表
saver.restore(sess, '/tmp/model.ckpt')
saver.restore(sess, '/tmp/model.ckpt-{}'.format(3))


#scope
##variable_scope
with tf.variable_scope('resnet'):
    v1 = tf.get_variable('v1', shape=[2,3,4]) #v1.name='resnet/v1',实际会有':0'后缀,可以不管
##共享变量
with tf.variable_scope('resnet'):
    v1 = tf.get_variable('v1', shape=[2,3,4], initializer=tf.random_normal_initializer())
with tf.variable_scope('resnet', reuse=True):
    v1 = tf.get_variable('v1')


#summary
##设置summary(返回值都是string)
c = tf.summary.scalar('name', a)
c = tf.summary.scalar('name', a, collections=[tf.GraphKeys.SUMMARIES]) #指定summary_op放到哪些collections
c = tf.summary.image('name', a, max_outputs=3)
c = tf.summary.histogram('name', a)
##op
summary_op = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
summary,... = sess.run([summary_op,...], feed_dict={...})
##writer
writer = tf.summary.FileWriter('/tmp/log')
writer.add_summary(summary, global_step=i)
writer.flush()
writer.close()
##tensor board
tensorboard --logdir=/tmp/mnist_logs/train/ -port=2213


##输入
c = tf.placeholder(tf.float32)
c = tf.placeholder(tf.float32, [None,224,224,1])
##常量(张量)
c = tf.constant([1,2,3])
c = tf.constant([1,2,3], dtype=tf.float32)
c = tf.constant(-1, shape=[2,3,3]) #相当于fill
c = tf.convert_to_tensor(val)
##变量
c = tf.Variable([2,3,4])
c = tf.get_variable('a', shape=[2,3], initializer=tf.constant_initializer(np.zeros((2,3))))


#学习器
##种类
opt = tf.train.AdamOptimizer(learn_rate)
opt = tf.train.GradientDescentOptimizer(learn_rate)
opt = tf.train.AdagradOptimizer(learn_rate)
opt = tf.train.AdadeltaOptimizer(learn_rate)
opt = tf.train.RMSPropOptimizer(learn_rate)
opt = tf.train.MomentumOptimizer(learn_rate)
opt = tf.train.FtrlOptimizer(learn_rate)
opt = tf.train.AdagradDAOptimizer(learn_rate)
##直接
train_op = opt.minimize(loss)
##分离求梯度
grads = tf.gradients(loss, variables)
train_op = opt.apply_gradients(zip(grad,variables))

#初始化
init_op = tf.global_variables_initializer()
init_op = tf.local_variables_initializer()


#初始化器
##常数
tf.constant_initializer([2,3,4])
tf.zeros_initializer()
tf.ones_initializer()
##传统随机分布
tf.random_uniform_initializer(minval=0, maxval=1)
tf.random_normal_initializer(mean=0.0, stddev=1.0)
##xavier
tf.glorot_uniform_initializer()
tf.glorot_normal_initializer()
##截断
tf.truncated_normal_initializer(mean=0.0, stddev=1.0)



##赋值
update_op = tf.assign(a, b) #a=b
update_op = tf.assign_add(a, b) #a+=b
update_op = tf.assign_sub(a, b) #a-=b
##下标赋值
update_op = tf.scatter_update(a, 0, b) #a[0]=b
update_op = tf.scatter_update(a, [2,0,1], b) #a[v[i]]=b[i]
update_op = tf.scatter_add(a, 0, b) #a[0]+=b
update_op = tf.scatter_sub(a, 0, b) #a[0]-=b


#下标
c = a[4]
c = a[2:4]
c = a[2:5,4:12]
##gather
c = tf.gather(a, [0,2,3], axis=0) #在某个轴上取一些下标
##slice
c = tf.slice(a, beg, size) #a[2:5,4:12]就是tf.slice(a,[2,4],[3,8])


#生成(张量)
##填充型
c = tf.zeros([2,3,4], dtype=tf.float32)
c = tf.zeros_like(a, dtype=None)
c = tf.ones([2,3,4])
c = tf.ones_like(a)
c = tf.fill([2,3,4], 5)
##数列型
c = tf.range(n)
c = tf.range(start, limit, delta=1)
c = tf.linspace(start, stop, num) #等差数列,左闭右闭
##随机型
c = tf.random_uniform([32,32], minval=0.0, maxval=1.0)
c = tf.random_normal([32,32], mean=-1, stddev=0.35)
c = tf.truncated_normal([32,32]) #截尾的正态分布(截去2stddev以上)



#随机
##随机种子
tf.set_random_seed(seed)
##打乱
c = tf.random_shuffle(a) #只随机打乱最高维


#上下文
##session
tf.get_default_session()
sess.as_default()
##graph
tf.get_default_graph()
g.as_default()
##设置使用哪些设备
tf.device('/cpu:0')
tf.device('/gpu:0')


#控制流
##cond(if-else)
##func都是py函数
b = tf.cond(fa, true_func, false_func)
b = tf.cond(fa, lambda: tf.add(a,b), lambda: tf.add(a,-b))
##case(switch)
b = tf.case([(fa,fxa),(fb,fxb)], fx_default)
##while_loop(while)
tf.while_loop(fa, body_func, loop_vars) #body_func要求参数和返回值都是loop_vars

#py-func
c = tf.py_func(func, [a,b,c], tf.int32)


#arg-
c = tf.argmax(a, axis=0)
c = tf.argmax(a, output_type=tf.int64) #可以改成tf.int32
c = tf.argmin(a)

#特殊op
##no_op
train_op = tf.no_op()
##group
group_op = tf.group([op1,op2])
##identity
x2 = tf.identity(x) #类似深拷贝


#类型转换
##数值间转换
c = tf.cast(a, tf.float32)
##str->num
c = tf.string_to_number(a)
c = tf.string_to_number(a, out_type=tf.float32) #还能选float64和int32,int64
##num->str
c = tf.as_string(a) #支持一般数字和bool



#矩阵
##矩阵乘法
c = tf.matmul(a, b)
##行列式
c = tf.matrix_determinant(a)
##矩阵的逆
c = tf.matrix_inverse(a)
##矩阵的迹
c = tf.trace(a)
##生成对角线矩阵
c = tf.diag(a) #a是对角线向量,c是矩阵



#规约
##sum
c = tf.reduce_sum(a) #规约到一个元素
c = tf.reduce_sum(a, axis=2) #沿着某轴规约
c = tf.reduce_sum(a, keep_dims=False) #规约后是否保持原shape
##其他
c = tf.reduce_mean(a)
c = tf.reduce_prod(a)
c = tf.reduce_min(a)
c = tf.reduce_max(a)
c = tf.reduce_all(a) #输入必须是bool
c = tf.reduce_any(a)


#字符串
##拼接
c = a + b
##split
c = tf.string_split(a, '/').values
##join
c = tf.string_join([a,b], '/')


#属性(张量)
c = tf.shape(a)
c = tf.size(a) #总元素个数,prod(shape)
c = tf.rank(a) #len(shape)


#队列
q = tf.FIFOQueue(capacity=100, dtypes=[tf.string,tf.int32], shapes=[[],[2,3]])
enq_op = q.enqueue([x1,x2])
enq_op = q.enqueue_many([x1s,x2s]) #x1s.shape=[None],x2s.shape=[None,2,3]
deq_op = q.dequeue()
q.is_closed()
q.size()


#batch
##batch
images,labels = tf.train.batch(
    [img,label],
    batch_size=64,
    capacity=2000,
    num_threads=4,
    allow_smaller_final_batch=True #假如前面设了epoch,这里不设,又不整除,最后会报异常
)
##shuffle_batch
images,labels = tf.train.shuffle_batch(
    [img,label],
    batch_size=64,
    capacity=2000,
    min_after_dequeue=500, #队列里最小的长度,用来保证shuffle
    num_threads=4,
    allow_smaller_final_batch=True
)


#数学函数
##四则运算
c = tf.add(a, b)
c = tf.subtract(a, b)
c = tf.multiply(a, b)
c = tf.div(a, b)
c = tf.mod(a, b)
##指数对数
c = tf.exp(a)
c = tf.log(a) #ln(a)
c = tf.pow(a, b)
##三角函数
c = tf.sin(a)
c = tf.cos(a)
c = tf.tan(a)
c = tf.asin(a)
c = tf.acos(a)
c = tf.atan(a)
##平方开方
c = tf.sqrt(a)
c = tf.square(a) #a^2
c = tf.rsqrt(a) #1/sqrt(a)
##基础数学符号
c = tf.neg(a) #-a
c = tf.inv(a) #1/a
##基础逻辑符号
c = tf.logical_and(a, b)
c = tf.logical_or(a, b)
c = tf.logical_not(a)
c = tf.logical_xor(a, b)
c = tf.equal(a, b)
c = tf.not_equal(a, b)
c = tf.less(a, b)
c = tf.less_equal(a, b)
c = tf.greater(a, b)
c = tf.greater_equal(a, b)
##分段函数
c = tf.abs(a)
c = tf.ceil(a)
c = tf.floor(a)
c = tf.round(a)
c = tf.maximum(a, b)
c = tf.minimum(a, b)
c = tf.sign(a)
##复数运算
c = tf.complex(a, b) #a+bi
c = tf.complex_abs(a) #|a|
c = tf.conj(a) #返回a的共轭
c = tf.real(a) #返回实部
c = tf.imag(a) #返回虚部


#类属性(值)和方法
##张量
a.device
a.dtype
a.graph
a.name
a.op
a.shape
a.set_shape([2,3]) #能将none去掉?
##变量
d.device
d.dtype
d.graph
d.initial_value #张量
d.initializer #op
d.name
d.shape
d.load(val, sess)
##张量shape
shape.as_list() #变成list(int)形式
shape.ndims #维度数
shape[0].value #变成int形式



#数据类型
##整数
tf.int8,tf.uint8,tf.uint16,tf.int16,tf.int32,tf.int64
##浮点数
tf.float16,tf.float32,tf.float64
##复数
tf.complex64,tf.complex128
##bool
tf.bool
##字符串
tf.string

#tf.GraphKeys
##OP
INIT_OP
LOCAL_INIT_OP
READY_FOR_LOCAL_INIT_OP
READY_OP
SUMMARY_OP
TRAIN_OP
UPDATE_OPS
##variable
VARIABLES
GLOBAL_VARIABLES
LOCAL_VARIABLES
MODEL_VARIABLES
CONCATENATED_VARIABLES
TRAINABLE_RESOURCE_VARIABLES
TRAINABLE_VARIABLES
WEIGHTS
BIASES
MOVING_AVERAGE_VARIABLES
##step
EVAL_STEP
GLOBAL_STEP
##summary
SUMMARIES
##saver
SAVEABLE_OBJECTS
SAVERS
##loss
LOSSES
REGULARIZATION_LOSSES

ACTIVATIONS
ASSET_FILEPATHS
COND_CONTEXT
LOCAL_RESOURCES
QUEUE_RUNNERS
RESOURCES
TABLE_INITIALIZERS
WHILE_CONTEXT


#设置哪些gpu对tf可见
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' #gpu0和gpu1可见


#改变梯度
@tf.RegisterGradient("QuantizeGrad")
def sign_grad(op, grad):
    return tf.clip_by_value(tf.identity(grad), -1, 1)

def binary(input):
    x = input
    with tf.get_default_graph().gradient_override_map({"Sign":'QuantizeGrad'}):
        x = tf.sign(x)
    return x


#多GPU-avg_grad
##变量定义一次,张量定义多次,梯度取均值
variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
all_grads = tf.get_collection('all_grads')
avg_grads = []
for grads_per_var in zip(*all_grads):
    grads_per_var = tf.stack(grads_per_var, axis=0)
    avg_grad = tf.reduce_mean(grads_per_var, axis=0)
    avg_grads.append(avg_grad)
train_op = opt.apply_gradients(zip(avg_grads, variables))

#keras-多gpu-网络定义
##利用reuse,保证name一样
with tf.variable_scope('resnet', reuse=(gpu_id!=0)):
    x = Conv2D(20,(5,5),name='conv1')(x)


#tf定义命令行参数
flags = tf.flags
flags.DEFINE_string("name", "default_val", "help")
flags.DEFINE_bool("name", "default_val", "help")
flags.DEFINE_float("name", "default_val", "help")
flags.DEFINE_integer("name", "default_val", "help")

FLAGS = flags.FLAGS

def main(_):
    print(FLAGS.name)

if __name__ == '__main__':
    tf.app.run()


#滑动平均
ema = tf.train.ExponentialMovingAverage(decay)
maintain_op = eme.apply([v1,v2]) #自动创建对应的滑动平均变量
...
sess.run(maintain_op) #moving_v1 = decay * moving_v1 + (1-decay) * v1
sess.run(eme.average(v1)) #moving_v1
#指数衰减
lr = tf.train.exponential_decay(init_lr, global_step, decay_steps, decay_rate, staircase=False)
##lr = init_lr * decay_rate^(global_step/decay_steps)
##如果staircase是true,那么global_step/decay_steps是整数除法,实现阶梯指数衰减


#tf-record
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

##write
writer = tf.python_io.TFRecordWriter(save_path)

example = tf.train.Example(features=tf.train.Features(feature={
    'img':_bytes_feature(img.tostring()),
    'label':_int64_feature(label)
}))
writer.write(example.SerializeToString())

writer.close()

##read
file_queue = tf.train.string_input_producer(['a.rcd','b.rcd'], num_epochs=10) #这实际是一个队列
reader = tf.TFRecordReader()
_, serialized_example = reader.read(file_queue)
features = tf.parse_single_example(serialized_example, features={
    'img':tf.FixedLenFeature([], tf.string),
    'label':tf.FixedLenFeature([], tf.int64)
})
img = features['img']
img = tf.decode_raw(img, tf.uint8) #img前面tostring了
img = tf.reshape(img, [224,224,3]) #img在tostring的时候丢失了shape信息
label = features['label']
images,labels = tf.train.batch(
    [img,label],
    batch_size=64,
    capacity=3000,
    num_threads=4
)

##run
sess.run(tf.local_variables_initializer()) #设了epoch,就一定要有这句
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord,sess=sess)
##读完之后,会抛出异常
coord.request_stop()
coord.join(threads)




'''
tf.nn.embedding_look
tf.nn.nce_loss
rnn
tf.tuple
'''