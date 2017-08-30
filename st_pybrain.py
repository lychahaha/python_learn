#coding=utf-8
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import random

net = buildNetwork(2,3,1,bias=True)

ds = SupervisedDataSet(2,1)
for i in range(100):
	a = random.randint(0,1)
	b = random.randint(0,1)
	ds.addSample((a, b), (a | b))

trainer = BackpropTrainer(net, ds)

while True:
	error_rate = trainer.train()
	print error_rate
	if error_rate < 0.05:
		break

for i in range(20):
	a = random.randint(0,1)
	b = random.randint(0,1)		
	ret = net.activate([a, b])
	c = 1 if ret[0] > 0.5 else 0
	print "%d | %d = %d" % (a, b, c)