#coding=utf-8
import json

#基础转化
#obj->str
json.dumps(obj)
json.dumps(obj, sort_keys=True)#排序
json.dumps(obj, indent=4)#设置缩进
json.dumps(obj, separators)#去掉空格等
#str->obj
json.loads(string)


#带自定义数据类型的转化

#1.把obj变成dict先
#obj->str
json.dumps(obj, default=obj2dict)
#str->obj
json.loads(obj, object_hook=dict2obj)

#2.继承
class encoder(json.JSONEncoder):
	def default(self, obj):
		#convert obj to dict
		d = {}
		d['__class__'] = obj.__class__.__name__
		d['__module__'] = obj.__module__
		d.update(obj.__dict__)
		return d

class decoder(json.JSONDecoder):
	def __init__(self):
		json.JSONDecoder.__init__(self, object_hook=self.dict2obj)

	def dict2obj(self, d):
		#convert dict to obj
		if '__class__' in d:
			class_name = d.pop('__class__')
			module_name = d.pop('__module__')
			module = __import__(module_name)
			class_ = getattr(module, class_name)
			args = dict((key.encode('ascii'), value) for key, value in d.items())
			inst = class_(**args)
		else:
			inst = d	
		return inst		

d = encoder().encode(obj)
obj = decoder().decode(d)		