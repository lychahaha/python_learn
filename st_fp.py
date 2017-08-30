#map(func, seq[,seq,...])
#seq里每个元素作为参数传入func,返回值作为元素的新值
map(func, seq)

#reduce(func, seq[,init])
#func是一个二元运算函数,seq会pop出前两个元素由func计算出结果放到seq的开头,不断迭代直到len(seq)==1
#有init时相当于init放进seq的开头
reduce(func, seq)

#filter(func, seq)
#func以seq每个元素为参数,返回bool.如果true则在seq保留该元素
filter(func, seq)

#zip(seq[,seq,...])
#压缩[a,b,c][1,2,3]->[(a,1),(b,2),(c,3)]
zip(seq1, seq2)
#zip(*seq)
#解压,压缩的逆过程
zip(*seq)