#jh:class-base,lazy-type
#coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


s.values
df.values


#初始化
##series
s = pd.Series([1,2,3])
s = pd.Series({'a':3,'b':4,'c':5}) #index->value
##np
df = pd.DataFrame(np.ones((4,3)), index=['A','B','C','D'], columns=['a','b','c'])
##dict of "list"
df = pd.DataFrame({
    'a':[1,2,3],
    'b':pd.Series([2,3,4]),
}, index=['r1','r2','r3'])
##list of dict
df = pd.DataFrame([
    {'c1':1,'c2':2,},
    {'c1':3,'c2':4,},
], index=['r1','r2'])


#数据合并
##append
df3 = df1.append(df2) #按行append
##concat
df3 = pd.concat([df1,df2]) #按行concat
df3 = pd.concat([df1,df2], axis=1) #按列concat
##merge
'''
how: inner|left|right|outer
on: 默认是两个df的列交集
left_on,right_on: 左右两个df的连接键分别用不同的名字
'''
df3 = pd.merge(df1, df2, how='inner', on=None, left_on=None, right_on=None)
df3 = pd.merge(df1, df2, on='pid') #经典内连接
df3 = pd.merge(df1, df2, left_on='pid', right_on='id') #连了之后左右连接键都会保留
##join


#删除
df2 = df.drop('r1')
df2 = df.drop(['c1','c2'], axis=1)


#索引
##loc(行标签(切片/list)/行mask,列标签(切片/list)/列mask)
df.loc['r1','c1']
df.loc[['r2','r3'],['c2','c5']]
df.loc['r1']
df.loc[df.A>0.5]
df.loc['r2':'r4'] #左闭右闭
##iloc(行下标(切片/list)/行mask,列下标(切片/list)/列mask)
df.iloc[0,2]
df.iloc[2:5] #左闭右开
df.iloc[[True,False,True]] #不能是Series
##[](列标签(list)/行mask)
###不建议使用[]赋值
df['c1']
df[['c1','c2']]
df[df.A>0.5]
##ix
###不要使用ix

#index/columns
df.index
df.columns
##ix->label
df.index[1]
df.columns[1]
##label->ix
df.index.get_loc('r1')
df.columns.get_loc('c1')
##重命名
df2 = df.rename(index=lambda x:x+1)
df2 = df.rename(columns={'old':'new'}) #单独改某列

#load
df = pd.read_csv('a.csv', sep=',')
df = pd.read_json('a.json')
df = pdf.read_pickle('a.pkl')
#save
df.to_cvs('a.csv')
df.to_json('a.json')
df.to_pickle('a.pkl')

#规约函数
##min
df2 = df.min() #把行规约
df2 = df.min(axis=1) #把列规约
##else
df2 = df.min()
df2 = df.max()
df2 = df.mean()
df2 = df.var()
df2 = df.median()
df2 = df.count() #非空行数
df2 = df.corr() #列与列的相关系数矩阵
df2 = df.describe() #一系列统计信息(count,mean,std,min,25%,50%,75%,max)

#聚合
gb = df.groupby(['c1','c2'])
gb = df.groupby(['r1','r2'], axis=1)
##规约
gb.mean()
gb['c3'].mean()
##
gb.size()
##迭代
list(gb)
for keys,df2 in gb:
    keys #tuple,放着共同项(c1,c2)
    df2 #df,所有共同项的所有行

#排序
df2 = df.sort_values('c1')
df2 = df.sort_values('c1', ascending=False) #降序
df2 = df.sort_values(['c1','c2'], ascending=[True,False]) #多key
df2 = df.sort_values('r1', axis=1) #对列排序


#info
df2 = df.head(n) #查看前n行
df2 = df.tail(n) #查看后n行
df.info() #查看索引,数据类型,内存等信息


#缺失值
pd.NaT
np.nan
None
##判断缺失值
mask = df.isna()
mask = df.notna()
##去掉缺失值
df2 = df.dropna() #删除包含缺失值的行
df2 = df.dropna(axis=1) #删除包含缺失值的列
df2 = df.dropna(how='all') #整行都缺失才删除(默认是any)
##填充缺失值
df2 = df.fillna(1)
df2 = df.fillna(value={'c1':0,'c2':3}) #不同列给不同填充值



df2 = df.T



s2 = s.str.lower()
s2 = s.str.strip()
