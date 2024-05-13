from lxml import etree

# 载入html
html_str = '<a>a</a>'
tree = etree.HTML(html_str)

# xpath搜索
eles = tree.xpath('//a')
ele = eles[0]

# 亲戚元素
ele.getchildren() #儿子(一个list)
ele.getparent() #父亲
ele.getnext() #弟弟(没有就返回none)
ele.getprevious() #哥哥(没有就返回none)

# 元素属性
ele.text #获取文本
ele.attrib #所有属性的字典
ele.get('id') #获取某个key的value
ele.keys() #字典的keys
ele.values() #字典的values

# 删除元素
ele.remove(ele_son) #只能是爸爸删除儿子

# 获取元素HTML
etree.tounicode(ele, method='html')




# xpath
//input #查找所有的input

//input[@class="as bs"] #带属性筛选（属性必须全部写出来）
//input[contains(@class,"as")] #包含某属性
//input[contains(@class,"as") or contains(@class,"as")] #支持与或运算
//input[@td] #包含该属性的属性筛选
//*[@class="haha"] #符合某些属性的任意元素

//div[@id="e"]/input #该div的input儿子
//div[@id="e"]/input[1] #该div的所有input儿子的第一个
//div[@id="e"]/input[last()] #该div的所有input儿子的最后一个
//div[@id="e"]/* #该div的所有儿子（*其实是通配符）
//div[@id="e"]//div #该div的所有div子孙
//div[@id="e"]/.. #该div的父亲
//div[@id="e"]/preceding-sibling::p[1] #该div的前一个哥哥p
//div[@id="e"]/following-sibling::p[1] #该div的后一个弟弟p
//div[@id="e"]/following-sibling::*[1] #该div的后一个弟弟
//div[@id="e"]/ancestor::*[1] #该div的最老祖先

.//div #以当前元素为根搜索所有div子孙
./div #当前元素的所有div儿子
./.. #当前元素的父亲(./表示当前元素)

//div/a | //div/b #并集

//a/@href #提取所有a的href属性
