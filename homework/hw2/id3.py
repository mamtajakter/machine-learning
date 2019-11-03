#!/usr/bin/python
#
# CIS 472/572 -- Programming Homework #1
#
# Starter code provided by Daniel Lowd, 1/25/2018
#
#

from __future__ import division
from math import log


import math
from collections import Counter


import sys
sys.setrecursionlimit(10000)

import re
# Node class for the decision tree
import node


train=None
varnames=None
test=None
testvarnames=None
root=None
posORneg=-1

# Helper function computes entropy of Bernoulli distribution with
# parameter p
def entropy(p):
	r=0
	if p>0 and p<1:
		r+=p*log(p,2)
		pi=1-p
		r+=pi*log(pi,2)
	else:
		r+=0

	r*=-1
	
	return r

# Compute information gain for a particular split, given the counts
# py_pxi : number of occurences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of ocurrences of y=1
# total : total length of the data
def infogain(py_pxi, pxi, py, total):

	if (py==total) or (total-py==0):
		s=0
	else:
		s=entropy(float(py)/total)

	if (py_pxi==pxi) or (pxi-py_pxi==0):
		p1=0
	else:
		p1=entropy(float(py_pxi)/pxi)

	if ((py-py_pxi)==(total-pxi)) or (((total-pxi)-(py-py_pxi))==0):
		p2=0
	else:
		p2=entropy(float(py-py_pxi)/(total-pxi))

	g=s-(float(pxi)/total)*p1-(float(total-pxi)/total)*p2
	#print(g)
	return g

# OTHER SUGGESTED HELPER FUNCTIONS:
# - collect counts for each variable value with each class label
# - find the best variable to split on, according to mutual information
# - partition data based on a given variable



# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
    	data.append([int(x) for x in p.split(l.strip())])
    return (data, varnames)

# Saves the model to a file.  Most of the work here is done in the
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
    f = open(modelfile, 'w+')
    root.write(f, 0)

# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def py_total_cal(data):

	total=0
	py=0
	i=0
	for i in range(len(data)):
		total +=1
		if data[i][-1]==1:
			py+=1
	
	return (py,total)

def pypxical(data,varnames,i):
	gain=0
	g=0
	pxi=0
	py_pxi=0
	for j in range (len(data)):
		pxi+=data[j][i]
		if (data[j][i]+data[j][-1])==2:
			py_pxi+=1
	return (py_pxi,pxi)

def selectMaxGainAttribute(data,varnames,alreadyConsidered):
	(py,total)=py_total_cal(data)
	##py_total_cal
	i=0
	initial_gain=0.0
	root_index=0
	for i in range (len(data[0])-1):
		if i not in alreadyConsidered:
			(py_pxi,pxi)=pypxical(data,varnames,i)
			gain=infogain(py_pxi, pxi, py, total)
			if gain>initial_gain:
				initial_gain=gain
				root_index=i

	return (root_index,gain)  # returns the column_number which is best	

def posData(data,varnames):

	positiveDataSet=[]
	
	i=0
	j=0
	for j in range (len(data)):
		if data[j][varnames]==1:
			positiveDataSet.append(data[j])
		
	return positiveDataSet

def negData(data,varnames):

	negativeDataSet=[]

	i=0
	j=0
	for j in range (len(data)):
		if data[j][varnames]==0:
			negativeDataSet.append(data[j])

	return negativeDataSet

def checkAllSame(data,varnames):
	is_one= 0 
	is_zero = 0 

	for i in data:
		if i[-1] == 1:
			is_one+=1
		else:
			is_zero+=1
	if is_one>is_zero:
		return (is_one,1)
	else:
		return (is_zero,0)
	
	# 	A	B	C	D	[A_and_B]_or_[C_and_D]
	# 	0	0	0	0	0
	# 	0	0	0	1	0
	# 	0	0	1	0	0
	# 	0	0	1	1	1
	# 	0	1	0	0	0
	# 	0	1	0	1	0
	# 	0	1	1	0	0
	# 	0	1	1	1	1
	# 	1	0	0	0	0
	# 	1	0	0	1	0
	# 	1	0	1	0	0
	# 	1	0	1	1	1
	# 	1	1	0	0	1
	# 	1	1	0	1	1
	# 	1	1	1	0	1
	# 	1	1	1	1	1
def build_tree(data, varnames):  

	bestChosenAttribute=-1
	alreadyConsidered=[]
	leftToBeConsidered=[]
	global_varnames=globals().get('varnames')

	#print("varnames")
	#print(varnames)

	#For the first time varnames is ['A', 'B', 'C', 'D', '[A_and_B]_or_[C_and_D]']
	#for the next recursions: varnames is leftToBeConsidered

	(x,value)= checkAllSame(data,varnames)
	if x==len(data):
		#print("Leaf 1")
		return node.Leaf(global_varnames,value)

	if (len(varnames) == 0):
		#print("Leaf 2")
		return node.Leaf(global_varnames,value)

	
	if len(data[0])==len(varnames):		
		j=0
		for j in range (len(varnames)):
			leftToBeConsidered.insert(j,j)
		alreadyConsidered.append(len(varnames)-1)  #For the first time: alreadyConsidered is [4]

		#print("first time")
	else:
		leftToBeConsidered.extend(varnames)
		j=0
		for j in range(0, leftToBeConsidered[len(varnames)-1]):
			if j not in leftToBeConsidered:
				alreadyConsidered.append(j)#For the next recursions: alreadyConsidered is [0]--[0, 2]--[0]--[0, 1]--[0, 1, 2]
		#print("I am recursing")

	#print("alreadyConsidered")
	stri=''
	for row in alreadyConsidered:
		stri=stri+','+ global_varnames[row]
	#print(stri)



	if len(leftToBeConsidered)==1:
		#print("Leaf 3")
		return node.Leaf(global_varnames,value)

	(bestChosenAttribute,maxGain)=selectMaxGainAttribute(data,varnames,alreadyConsidered)# bestChosenAttribute: 0 -- 2--3--1--2--3

	if bestChosenAttribute in leftToBeConsidered:
		leftToBeConsidered.remove(bestChosenAttribute)#leftToBeConsidered: [1, 2, 3, 4]--[1, 3, 4]--[1, 4]--[2, 3, 4]--[3, 4]--[4]


	#print("bestChosenAttribute")
	#print(global_varnames[bestChosenAttribute])

	#print("Gain")
	#print(maxGain)

	#print("leftToBeConsidered")
	stri=''
	for row in leftToBeConsidered:
		stri=stri+','+ global_varnames[row]
	#print(stri)


	if len(data)==0:

		return node.Leaf(global_varnames,value)

	negativeDataSet=negData(data,bestChosenAttribute)
	positiveDataSet=posData(data,bestChosenAttribute)

	if maxGain==0.0 and bestChosenAttribute>=0:
		#print("Leaf 4")
		if data[0][-1]==0:
			return node.Split(globals().get('varnames'),bestChosenAttribute,node.Leaf(global_varnames,negativeDataSet[0][-1]),build_tree(positiveDataSet,leftToBeConsidered))
		else:
			return node.Split(globals().get('varnames'),bestChosenAttribute,build_tree(negativeDataSet,leftToBeConsidered),node.Leaf(global_varnames,positiveDataSet[0][-1]))
		
	elif maxGain>0:
		return node.Split(globals().get('varnames'),bestChosenAttribute,build_tree(negativeDataSet,leftToBeConsidered),build_tree(positiveDataSet,leftToBeConsidered))#for the next recursions: varnames is leftToBeConsidered

	else:
		#print("Leaf 5")
		return node.Leaf(global_varnames,value)


# "varnames" is a list of names, one for each variable
# "train" and "test" are lists of examples.
# Each example is a list of attribute values, where the last element in
# the list is the class value.
def loadAndTrain(trainS,testS,modelS):
	global train
	global varnames
	global test
	global testvarnames
	global root
	(train, varnames) = read_data(trainS)
	(test, testvarnames) = read_data(testS)
	modelfile = modelS

	# build_tree is the main function you'll have to implement, along with
    # any helper functions needed.  It should return the root node of the
    # decision tree.
	root = build_tree(train, varnames)
	print_model(root, modelfile)

def runTest():
	correct = 0
	# The position of the class label is the last element in the list.
	yi = len(test[0]) - 1
	for x in test:
		# Classification is done recursively by the node class.
        # This should work as-is.
		pred = root.classify(x)
		if pred == x[yi]:
			correct += 1
	acc = float(correct)/len(test)
	return acc


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 3):
    	print ('Usage: id3.py <train> <test> <model>')
    	sys.exit(2)
    loadAndTrain(argv[0],argv[1],argv[2])

    acc = runTest()
    print ("Accuracy: ",acc)

if __name__ == "__main__":
    main(sys.argv[1:])
