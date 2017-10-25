#jh:mix-base,lazy-type
import tensorflow as tf
import numpy as np


#keras.layers
from tensorflow.contrib.keras.python.keras.layers import Conv2D,Dense,MaxPooling2D,ZeroPadding2D,Flatten,BatchNormalization,Dropout,AveragePooling2D,Activation
from tensorflow.contrib.keras.python.keras.layers import Conv3D,MaxPooling3D,AveragePooling3D,ZeroPadding3D
from tensorflow.contrib.keras.python.keras import layers
##conv
##变量放在TRAINABLE_VARIABLES
x = Conv2D(64, (3,3), strides=(1,1), padding='valid', name='conv1')(x) #w:'conv1/kernel', b:'conv1/bias'
x = Conv2D(32, (3,3), padding='same', activation='relu')(x)
##fc
##变量放在TRAINABLE_VARIABLES
x = Dense(256, activation='relu', name='fc1')(x) #w:'fc1/kernel', b:'fc1/bias'
##pool
x = MaxPooling2D((2,2), strides=(2,2))(x)
x = AveragePooling2D((2,2), strides=(2,2))(x)
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
##补零
x = ZeroPadding2D((3,3))(x) #两边都加3
##3D
x = Conv3D(32, (3,3,3), strides=(2,2,2), padding='same', name='conv1')(x)
x = MaxPooling3D((2,2,2), strides=(2,2,2))(x)
x = AveragePooling3D((2,2,2), strides=(2,2,2))(x)
x = ZeroPadding3D((3,3,3))(x)


#sess
sess = tf.Session()
##带config
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True #按需增长
sess_config.allow_soft_placement=True #设备不合适时,自动寻找合适的设备
sess_config.log_device_placement=True
sess = tf.Session(config=sess_config)
##可以支持eval的
sess = tf.InteractiveSession()
##运行
ans = sess.run(obj, feed_dict={a1:b1,a2:b2})
##关闭
sess.close()


#收集
##add
tf.add_to_collection('loss', a)
##get
l = tf.get_collection('loss')
l = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='conv')
##tf.GraphKeys
tf.GraphKeys.TRAINABLE_VARIABLES
tf.GraphKeys.UPDATE_OPS
tf.GraphKeys.SUMMARIES


##输入
c = tf.placeholder(tf.float32)
c = tf.placeholder(tf.float32, [None,224,224,1])
##常量
c = tf.constant([1,2,3])
c = tf.constant([1,2,3], dtype=tf.float32)
c = tf.constant(-1, shape=[2,3,3]) #相当于fill
##变量
c = tf.Variable([2,3,4])
c = tf.get_variable('a', shape=[2,3], initializer=tf.constant_initializer(1))
#张量
c = tf.convert_to_tensor(val)


#学习器
##种类
opt = tf.train.AdamOptimizer(learn_rate)
opt = tf.train.GradientDescentOptimizer(learn_rate)
##直接
train_op = opt.minimize(loss)
##分离求梯度
grads = tf.gradients(loss, variables)
train_op = opt.apply_gradients(zip(grad,variables))


#初始化器
init_op = tf.global_variables_initializer()
tf.constant_initializer(1)
tf.zeros_initializer()
tf.ones_initializer()
tf.random_uniform_initializer(minval=0, maxval=1)
tf.random_normal_initializer(mean=0.0, stddev=1.0)
tf.truncated_normal_initializer(mean=0.0, stddev=1.0)


#保存与载入
##saver
saver = tf.train.Saver()
saver = tf.train.Saver([a,b,c]) #指定要保存的东西
##save
save_path = saver.save(sess, '/tmp/model.ckpt')
save_path = saver.save(sess, '/tmp/model.ckpt', global_step=3)
##load
saver.restore(sess, '/tmp/model.ckpt')
saver.restore(sess, '/tmp/model.ckpt-{}'.format(3))


#共享变量
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
writer.close()


##张量属性(值)
a.device
a.dtype
a.graph
a.name
a.op
a.shape
##变量属性(值)和方法
d.device
d.dtype
d.graph
d.initial_value #张量
d.initializer #op
d.name
d.shape
d.load(val, sess)


#tensor<->list(tensor)
##list->tensor
t = tf.stack(l, axis=0) #创建新轴
c = tf.concat([a,b,c], 2) #不创建新轴
##tensor->list
l = tf.unstack(t, axis=0)
##n个张量相加
t = tf.add_n(l)


#改变形状
##reshape
c = tf.reshape(a, [2,3,4])
##去掉长度为1的维度
c = tf.squeeze(a) #去掉所以
c = tf.squeeze(a, axis=[1,4]) #去掉指定的
##插入长度为1的维度
c = tf.expand_dims(a, axis=2)
##转置
c = tf.transpose(a)
c = tf.transpose(a, perm=[3,0,1,2])


#赋值
update_op = tf.assign(a, b) #a=b
update_op = tf.assign_add(a, b) #a+=b
update_op = tf.assign_sub(a, b) #a-=b


#下标
c = a[4]
c = a[2:4]
c = a[2:5,4:12]

#随机
##随机张量
c = tf.random_uniform([32,32], minval=0.0, maxval=1.0)
c = tf.random_normal([32,32], mean=-1, stddev=0.35)
c = tf.truncated_normal([32,32]) #截尾的正态分布(截去2stddev以上)
##随机种子
tf.set_random_seed(seed)
##打乱
c = tf.random_shuffle(a) #只随机打乱最高维


#设备
tf.device('/cpu:0')
tf.device('/gpu:0')


#条件
#func都是py函数
b = tf.cond(fa, true_func, false_func)
b = tf.cond(fa, lambda: tf.add(a,b), lambda: tf.add(a,-b))
b = tf.cond(fa, true_func, false_func) #是否让


#py-func
c = tf.py_func(func, [a,b,c], tf.int32)


#arg-
c = tf.argmax(a, axis=0)
c = tf.argmax(a, output_type=tf.int64) #可以改成tf.int32
c = tf.argmin(a)


#no_op
train_op = tf.no_op()


#类型转换
##数值间转换
c = tf.cast(a, tf.float32)
##str->num
c = tf.string_to_number(a)
c = tf.string_to_number(a, out_type=tf.float32) #还能选float64和int32,int64
##num->str
c = tf.as_string(a) #支持一般数字和bool


#复杂函数
##激活函数
x = tf.nn.softmax(x)
x = tf.nn.relu(x)
x = tf.nn.tanh(x)
x = tf.nn.sigmoid(x)
##loss
x = tf.nn.l2_loss(x)
##layer
x = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')
x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
x = tf.nn.dropout(x, keep_rate)
##top-k
x = tf.nn.in_top_k(logits, labels, k) #logits是二维的,labels是一维的
##onehot
c = tf.one_hot(a, cla_nums)


#生成
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


#属性
c = tf.shape(a)
c = tf.size(a) #总元素个数,prod(shape)
c = tf.rank(a) #len(shape)


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


#tf-record
writer = tf.python_io.TFRecordWriter(save_path)

context = tf.train.Features({
    feature={
        'video_id':tf.train.Feature(btypes_list=tf.train.BtypesList(value=[value]))
        'labels':tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    }
})
feature_lists = tf.train.FeatureLists(feature_list={
    'rgb':tf.train.FeatureList(feature=[tf.train.Feature(btypes_list=tf.train.BtypesList(value=[value])) for value in values ])
    'audio':tf.train.FeatureList(feature=[tf.train.Feature(btypes_list=tf.train.BtypesList(value=[value])) for value in values ])
})

example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
write.write(example.SerializeToString())

writer.close()

