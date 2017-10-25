#jh:class-base,lazy-type
#coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Series
#列表
s = pd.Series([1,2,'3'])
s2 = pd.Series([1,2,3], index=[0,1,2], dtype='float32')

#DataFrame
#数据表
date = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=date, columns=list('1234'))

df2 = pd.DataFrame({'A':np.array([1,2,3]),
					'B':pd.Series([1,2,3]),
					'C':pd.Categorical(['t','f','t']),
					'D':'abc'
					})

#列类型
df.dtypes

#索引
df.index

#表头(pd.index)
df.columns

#表的值(np.array)
df.values

#转置
df.T

#单独的列(pd.Series)
df['A']
df.A

#加一列
df['E'] = ['1','2','3']
#index要对应df的
df['F'] = pd.Series([1,2,3],index=[0,1,2])


#行切片
#根据行数
df[0:3]
#根据索引
df['20130102':'20130106']

#标签二维切片
#前面选行,后面选列(pd.DataFrame)
df.loc['20130102':'20130106',['A','B']]
#单行(pd.Series,列标签是索引,行数据是值)
df.loc['20130102',['A','B']]
#单行全部列(pd.Series,列标签是索引,行数据是值)
df.loc['20130102']
#单行单列(标量)
df.loc['20130102','A']

#下标二维切片
#单行(pd.Series,列标签是索引,行数据是值)
df.iloc[3]
#前面行,后面列(pd.DataFrame)
df.iloc[3:5,0:2]
#单行单列(标量)
df.iloc[1,1]
df.iat[1,1]

#布尔过滤
#某个列过滤
df[df.A>0]
#不满足条件的项变成NaN
df[df>0]
#包含
df[df['E'].isin(['1','2'])]

#去掉缺失值的行
df.dropna(how='any')
#填充缺失值
df.fillna(value=5)
#求缺失布尔矩阵
pd.isnull(df)

#头部行和尾部行
df.head()
df.tail(3)

#统计量汇总
df.describe()

#排序
#参数1是哪个轴(1表示纵轴),参数2是倒序
df.sort_index(axis=1, ascending=False)

#按列排序
df.sort(columns='B')#不推荐了
df.sort_values(by='B')


#深拷贝
df.copy()

#均值
df.mean(axis=0)