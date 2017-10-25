#jh:proc-base,lazy-type

#LEGB
#L:local
#E:enclosing
#G:global
#B:build-in




def set_passline(passline):
	def cmp(val):
		if val >= passline:
			print 'pass'
		else:
			print 'failed'
	return cmp

f_100 = set_passline(100)
f_150 = set_passline(150)

f_100(89)
f_150(89)




def my_sum(*arg):
	return sum(arg)

def my_avg(*arg):
	return sum(arg)/len(arg)

def deco(func):
	def in_deco(*arg):
		if len(arg) == 0:
			return 0
		for val in arg:
			if not isinstance(val, int):
				return 0
		return func(*arg)
	return in_deco

my_sum = deco(my_sum)
my_avg = deco(my_avg)

print my_sum(1,2,3,4,5)
print my_avg(1,2,3,4,5)




def deco(func):
	def in_deco(*arg):
		if len(arg) == 0:
			return 0
		for val in arg:
			if not isinstance(val, int):
				return 0
		return func(*arg)
	return in_deco

@deco
def my_sum(*arg):
	return sum(arg)

@deco
def my_avg(*arg):
	return sum(arg)/len(arg)

print my_sum(1,2,3,4,5)
print my_avg(1,2,3,4,5)




def deco(func):
	def in_deco(x, y):
		if y == 0:
			return 0
		return func(x, y)
	return in_deco

@deco
def myDiv(x, y)
	return x/y

print myDiv(6,2)
print myDiv(3,0)




