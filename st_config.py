import configparser

#typical
conf = configparser.ConfigParser()
conf.read("a.ini")
name = conf.get("section1", "name")

conf.set("section2", "port", "8081")
conf.write(open("a.ini", 'w'))


#full

#定义
conf = configparser.ConfigParser()

#读取文件
conf.read("a.ini")
#写文件
conf.write(open("a.ini", 'w'))

#读取值
name = conf.get("section1", "name")
port = conf.getint("section2", "port")
#写值
conf.set("section2", "port", "8081")

#获取所有section的名字
sections = conf.sections()
#获取某个section上所有键
ops = conf.options("section1")

#增加新section
conf.add_section("new_section")


#a.ini
'''
[section1]
name = xxx

[section2]
port = 8080

'''