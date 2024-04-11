# mysql,MariaDB

# MongoDB
## database-collection-document-key

# 列出所有数据库
show dbs
# 进入某个数据库
use dbname
#删除数据库
db.dropDatabase()

# 列出所有集合
show collections
# 创建集合
db.createCollection("cname")
#删除集合
db.cname.drop()

# 增加文档
db.cname.insert( {name:"haha",age:10} ) #document是json格式的dict
db.cname.insert( [{name:"haha",age:10},{name:"hehe",age:11}] ) #多个doc，外面则是json格式的list

# 查找文档
## find返回的是一个数组
db.cname.find() #查所有
db.cname.find( {name:"haha"} ) #指定属性筛选
db.cname.find( {name:"haha",age:11} ) #多个属性筛选（隐式的and）
db.cname.find( {age:{$gt:12}} ) #筛选age>12的文档($gt $gte $lt $ lte)
db.cname.find( {age:{$in:[12,14]}} ) #逻辑in
db.cname.find( {$or:[{age:11},{name:haha}]} ) #逻辑or
db.cname.find( {name:/^p/} ) #p开头的字符串

db.cname.find( {age:11}, {name:1} ) #筛选后只获取name字段，1表示要显示，0表示不显示（只有id需要显示写成0）

db.cname.find().count() #统计个数
db.cname.find().limit(10) #只显示前10条
db.cname.find().skip(10).limit(20) # 显示第11~20条
db.cname.find().sort( {age:1} ) #按age排序，1表示升序，-1表示降序

# 聚合(最外必须要有[])
db.cname.aggregate([  {$group:{_id:"date", count:{"$sum":1}}}  ])  #使用group聚合，按date作为key，生成count字段，该字段以sum函数进行聚合
# $push:将所有value group变成一个list

db.cname.aggregate([    $match:{name:"haha"}         {$group:{_id:"date", count:{"$sum":1}}}    ]) #先match（就是find操作）再group

db.cname.aggregate([   {$project:{age:1}}        ]) #投影操作（只显示该字段），1表示要显示，0表示不显示（只有id需要显示写成0）
db.cname.aggregate([   {$project:{fullname:{$concat:["$firstname"," ","$lastname"]}}}   ]) #合并字符串
db.cname.aggregate([   {$project:{aname:{$substr:["$name",2,5]}}}   ]) #子串（那2个数字分别为起始ix和长度）
db.cname.aggregate([   {$project:{aname:{$replaceAll:{input:"$name",find:"ha",replacement:"he"}}}}   ]) #字符串替换
db.cname.aggregate([   {$project:{aname:{$toLower:"$name"}}}   ]) #转小写（转大写是toUpper)


db.cname.aggregate([   {$sort:{age:1}}        ]) #排序操作，1表示升序，-1表示降序
db.cname.aggregate([   {$limit:2}        ]) #只显示前x条

db.cname.aggregate([   {$unwind:$children}        ]) #拆解，将list的value拆成多个doc
db.cname.aggregate([   {$unwind:{path:'$children',preserveNullAndEmptyArrays:true}}        ]) #保留没该字段或该字段为空list的doc

# 修改文档
db.cname.update(  {name:"haha"},   {$set:{age:14}}   ) #赋值修改
db.cname.update(  {name:"haha"},   {$currentDate:{lastTime:true}}    )  #将lastTime修改为当前日期，true表示如果没有该字段，则创建该字段

# 删除文档
db.cname.deleteMany( {"name":"haha"} ) #先筛选再删除
db.cname.deleteOne( {"name":"haha"} ) #只删除匹配的第一个
