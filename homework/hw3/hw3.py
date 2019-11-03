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

def makeDataSet(yz):
	train_data=[]
	randomRows=[]
	for i in xrange(100):
		train_data.append([])
		for j in xrange(100):
			train_data[i].append(1)
	i=0
	j=0
	for j in range (len(train_data[0])):#columns
		if j==0:
			for i in range(len(train_data)):
				if i<50:
					train_data[i][j]=0
				else:
					train_data[i][j]=1
		else:
			for i in range(len(train_data)):
				train_data[i][j]=randint(0, 1)
	return train_data

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

counter=0	
accuracies=[]
for k in range(10):
	train_data=[]
	test_data=[]
	train_data=makeDataSet ("Nothing")
	test_data=makeDataSet ("Nothing")
	distances=[]
	distances=makeDistances(train_data,test_data)
	minimumDistanceIndexes=[]
	minimumDistanceIndexes=getminimumDistanceIndexes(distances)
	counter=0
	i=0
	j=0
	for i in range(len(minimumDistanceIndexes)):
		j=minimumDistanceIndexes[i]
		if train_data[j][0]==test_data[i][0]:
			counter+=1
	accuracies.append(counter)
print(accuracies)
print reduce(lambda x, y: x + y, accuracies)/len(accuracies)
