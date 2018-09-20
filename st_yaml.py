import yaml

#str(f)->obj
obj = yaml.load(s)

#obj->str
s = yaml.dump(obj)

'''
字典:
a:
    b: 3
    c: 2

列表:
a:
    b: [2,3,3]
    c: [2,2,3]

数字会自动转换成数字
有复杂符号的字符串最好加引号

'''