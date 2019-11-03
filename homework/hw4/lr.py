# Mamtajs-MacBook-Pro:h mamtajakter$ Python3 lr.py spambase-train.csv spambase-test.csv 0.1 1.0 lrspam.txt
# Accuracy:  0.608
# Mamtajs-MacBook-Pro:h mamtajakter$ Python3 lr.py agaricuslepiotatrain1.csv agaricuslepiotatest1.csv 0.1 1.0 lrmushroom.txt
# Accuracy:  0.12188235294117647

#!/usr/bin/python
#
# CIS 472/572 - Logistic Regression Template Code
#
# Author: Daniel Lowd <lowd@cs.uoregon.edu>
# Date:   2/9/2018
#
# Please use this code as the template for your solution.
#
import sys
import re
from math import log
from math import exp
from math import sqrt
#from math import dot


MAX_ITERS = 100

# Load data from a file


def read_data(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  data = []
  header = f.readline().strip()
  varnames = p.split(header)
  namehash = {}
  for l in f:
    example = [int(x) for x in p.split(l.strip())]
    x = example[0:-1]
    y = example[-1]
    data.append( (x,y) )
  return (data, varnames)


def dot(w,x):

  return sum(m*n for m,n in zip(w,x))

def sigmoid( x, y, w, b):

  act = dot(w,x)+ b
  grad=0

  try:
    denom =  1+exp(y*act)
  except OverflowError:
    denom = float('inf')

  try:
    grad = 1/denom
  except ZeroDivisionError:
    grad = 0 

  return grad

# Train a logistic regression model using batch gradient descent
def train_lr(data, eta, l2_reg_weight):
  lamda=l2_reg_weight
  numvars = len(data[0][0])
  w = [0.0] * numvars
  b = 0.0
  weight_gradient=[0.0]*numvars
  sigma=0
  
  for j in range(MAX_ITERS):
    bias_gradient=0

    
    for (x,y)in data:
      sigma=sigmoid(x, y, w, b)
      
      for i in range(0,numvars):
        
        weight_gradient[i] += (x[i] * sigma * y)   

      bias_gradient += (sigma * y)

    magnitudes=0
    for i in range(0,numvars):
      magnitudes += pow(weight_gradient[i],2)

    magnitudes += pow(bias_gradient,2)
    magnitudes=sqrt(magnitudes)

    if magnitudes < 0.0001:
      break

    b+=(eta*bias_gradient)  

    for i in range(0,numvars):
      w[i]+=(eta*weight_gradient[i])-(eta*lamda*w[i])

  return (w,b)
  

# Predict the probability of the positive label (y=+1) given the
# attributes, x.

def predict_lr(model, x):

  (w,b) = model

  act = -(dot(w,x)+b)

  try:
    deno =exp(act)
  except OverflowError:
    deno = float('inf')

  try:
    prob = 1/deno
  except ZeroDivisionError:
    prob = 0 

  return prob


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
  if (len(argv) != 5):
    print('Usage: lr.py <train> <test> <eta> <lambda> <model>')
    sys.exit(2)
  (train, varnames) = read_data(argv[0])
  (test, testvarnames) = read_data(argv[1])
  eta = float(argv[2])
  lam = float(argv[3])
  modelfile = argv[4]

  # Train model
  (w,b) = train_lr(train, eta, lam)

  # Write model file
  f = open(modelfile, "w+")
  f.write('%f\n' % b)
  for i in range(len(w)):
    f.write('%s %f\n' % (varnames[i], w[i]))

  # Make predictions, compute accuracy
  correct = 0
  for (x,y) in test:
    prob = predict_lr( (w,b), x )
    #print(prob)
    if (prob - 0.5) * y > 0:
      correct += 1
  acc = float(correct)/len(test)
  print("Accuracy: ",acc)

if __name__ == "__main__":
  main(sys.argv[1:])
