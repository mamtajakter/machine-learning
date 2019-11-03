import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import glob
import random
import itertools
from scipy.spatial import distance
from random import randint



def random_gen(low, high):
    while True:
        yield random.randrange(low, high)


def getColumn(lst,i):
	return [column[i] for column in lst]


def makeTrainDataSet(yz):

	train_data=[
	[0.08,	0.15,	1,	0.1,	0.5],
[0.15,	0.19,	0.9,	0.25,	0.8],
[0.1,	0.35,	1,	0.05,	1],
[0.11,	0.4,	0.95,	0.2,	0.6],
[0.12,	0.1,	0.99,	0.3,	0.6],
[0.18,	0.15,	1,	0.2,	0.5]]

			# gen = random_gen(0, 99)
			# items = set()
			# for x in itertools.takewhile(lambda x: len(items) < 50, gen):
			# 	items.add(x)
			# for i in range(len(train_data)):
			# 	if i in items:
			# 		train_data[i][j]=0
			# 	else:
			# 		train_data[i][j]=1
	return train_data

def makeTestDataSet(yz):

	test_data=[
[0.03,	0.21,	1,	0.15,	0.7],
[0.14,	0.04,	1,	0.35,	0.5],
[0.13,	0.05,	1,	0.3,	0.3],
[0.06,	0.25,	0.94,	0.28,	0.9]]
			# gen = random_gen(0, 99)
			# items = set()
			# for x in itertools.takewhile(lambda x: len(items) < 50, gen):
			# 	items.add(x)
			# for i in range(len(train_data)):
			# 	if i in items:
			# 		train_data[i][j]=0
			# 	else:
			# 		train_data[i][j]=1
	return test_data


def makeDistances(train_data,test_data):
	distances=[]
	i=0
	j=0
	for i in range (len(test_data)):
		distances.append([])
		for j in range (len(train_data)):
			distances[i].append( distance.euclidean(test_data[i],train_data[j]))
	return distances





def getminimumDistanceIndexes(distances):
	return np.argmin(distances, axis=0)

	# i=0
	# j=0
	# minimumDistances=[]
	# currentColumn=[]


	# minDist=0

	# minimumDistanceIndexes=[]
	# for j in range(len(distances[0])):
	# 	currentColumn=getColumn(distances,j)
	# 	minimumDistances.append(min(currentColumn))
	# 	minDist=min(currentColumn)
	# 	for i in range (len(currentColumn)):
	# 		if minDist==currentColumn[i]:
	# 			minimumDistanceIndexes.append(i)
		
	# #print(minimumDistances)
	# print(len(minimumDistanceIndexes))
	# return minimumDistanceIndexes


counter=0	

accuracies=[]

for k in range(10):
	train_data=[]
	test_data=[]
	train_data=makeTrainDataSet ("Nothing")
	test_data=makeTestDataSet ("Nothing")

	distances=[]
	distances=makeDistances(train_data,test_data)

	df = pd.DataFrame(np.array(train_data).reshape(6,5), columns = list("abcde"))
	df.to_csv('training1.csv', sep=',')

	df = pd.DataFrame(np.array(test_data).reshape(4,5), columns = list("abcde"))
	df.to_csv('test1.csv', sep=',')

	df = pd.DataFrame(np.array(distances).reshape(6,4))
	df.to_csv('distances1.csv', sep=',')

	minimumDistanceIndexes=[]
	minimumDistanceIndexes=getminimumDistanceIndexes(distances)
	#print(minimumDistanceIndexes)
	counter=0
	i=0
	j=0
	#print(len(minimumDistanceIndexes))
	for i in range(len(minimumDistanceIndexes)):
		#print(minimumDistanceIndexes[i])
		j=minimumDistanceIndexes[i]
		if train_data[j][0]==test_data[i][0]:
			counter+=1
			#print(counter)

	accuracies.append(counter)


print(accuracies)
print reduce(lambda x, y: x + y, accuracies)/len(accuracies)
