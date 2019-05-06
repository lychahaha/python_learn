#jh:class-base,lazy-type
#coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

s = pd.Series([1,2,3])
s = pd.Series({'a':3,'b':4,'c':5}) #index->value

s.values
s.index

s['a']

df = pd.DataFrame(np.ones((4,3)), index=['A','B','C'], columns=['a','b','c'])
df = pd.DataFrame({
    'a':[1,2,3],
    'b':[2,3,4],
})

df.columns
df.index
df.values

df.a
df['a']
df.ix['i3']
df.ix['i3','a']
df.loc['a','i3']
df.iloc[2,3]
df[1:4]
df[df['a']>3]

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
##label->ix
df.index.get_loc('r1')
df.columns.get_loc('c1')

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



del df['a']
df2 = df.drop('a', axis=1)
df2 = df.drop(['i1','i2'])

df2 = df.T

mask = df.isnull()
mask = df.notnull()
mask = df.isna()
df2 = df.dropna(axis=1, how='all')
df2 = df.fillna(1)



s2 = s.str.lower()
s2 = s.str.strip()
