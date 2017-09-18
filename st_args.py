import argparse

#typical
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("name", help='xxx')
	parser.add_argument("-v", "--verbosity", help="xxx")

	args = parser.parse_args()
	print(args.v)

#full
if __name__ == '__main__':
	#定义参数列表
	parser = argparse.ArgumentParser()

	#定义定位参数(必填)
	#python a.py myname
	parser.add_argument("name", help='xxx')
	
	#定义名字参数(选填)
	#python a.py -v xxx
	#python a.py --verbosity=xxx
	parser.add_argument("-v", "--verbosity", help="xxx")

	##定义01名字参数(选填)
	#python a.py -v
	parser.add_argument("-k", action='store_true')

	##定义参数类型,默认值,值域
	parser.add_argument("-c", type=int)
	parser.add_argument("-d", type=int, choices=[1,2,3])
	parser.add_argument("-e", default='sss')

	##定义互斥参数
	group = parser.add_mutually_exclusive_group()
	group.add_argument("-a")
	group.add_argument("-b")

	#获取参数表
	args = parser.parse_args()

	#获取参数
	print(args.v)#没定义'--'名字时,才能使用
	print(args.verbosity)
	
	#参数类型
	#默认str类型
	#有定义类型的,按照定义的类型
	#01参数是bool类型

	#传递多值时,可以加双引号变成一个字符串,再在py里处理
	#python a.py -k "a b c"