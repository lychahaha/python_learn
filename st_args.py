#jh:proc-base-typical,key-type
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
	parser.add_argument("--verbosity", help="xxx")

	##定义01名字参数(选填)
	#python a.py -k
	parser.add_argument("-k", action='store_true')
	parser.add_argument("-k", action='store_false')
	#更一般的01名字参数
	parser.add_argument("-k", action='store_const', const=42)

	##定义参数类型,默认值,值域
	parser.add_argument("-c", type=int)
	parser.add_argument("-d", type=int, choices=[1,2,3])
	parser.add_argument("-e", default='sss')

	##定义互斥参数
	group = parser.add_mutually_exclusive_group()
	group.add_argument("-a")
	group.add_argument("-b")

	#多值参数
	##指定值个数
	##python a.py -s 1 2
	parser.add_argument("-s", nargs=2) 
	##不定值个数(至少一个)
	##python a.py -s 1 2 3 
	parser.add_argument("-s", nargs='+')
	##不定值个数(可以零个)
	##python a.py -s
	parser.add_argument("-s", nargs='*')
	##0个或1个值
	##python a.py -s 1
	parser.add_argument("-s", nargs='?')

	#换名字(为了调用的方便性和代码的可读性)
	parser.add_argument("-v", dest='version') #args.version

	#获取参数表
	args = parser.parse_args()

	#获取参数
	print(args.v)#没定义'--'名字时,才能使用
	print(args.verbosity)
	#可选参数没有时,值为None

	#参数类型
	#默认str类型
	#有定义类型的,按照定义的类型
	#01参数是bool类型

	#传递多值时,参数是个list