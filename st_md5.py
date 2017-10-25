#jh:proc-base-onlyone,key-type
#coding=utf-8
import md5

hash = md5.new()
hash.update('hahahaha')

print hash.hexdigest()

print md5.md5('hahahaha').hexdigest()


import sha

hash = sha.new()
hash.update('hahahaha')

print hash.hexdigest()

print sha.sha('hahahaha').hexdigest()