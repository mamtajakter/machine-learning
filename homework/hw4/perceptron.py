# Mamtajs-MacBook-Pro:h mamtajakter$ Python3 perceptron.py spambase-train.csv spambase-test.csv pspam.txt
# Accuracy:  0.871
# Mamtajs-MacBook-Pro:h mamtajakter$ Python3 perceptron.py agaricuslepiotatrain1.csv agaricuslepiotatest1.csv pmushroom.txt
# Accuracy:  0.8974117647058824

#!/usr/bin/python
#
# CIS 472/572 - Perceptron Template Code
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
    # Each example is a tuple containing both x (vector) and y (int)
    data.append( (x,y) )
  return (data, varnames)


def train_perceptron(data):
  
  numvars = len(data[0][0])
  
  w = [0.0] * numvars

  b = 0.0

  a = [0.0] * numvars
  act=0

  rows = len(data)

  for iter in range(0,MAX_ITERS):
    
    correct=0
    for d in data:

      (x,y)= d
      act = 0
      for i in range(0,numvars):
        act = act + (w[i]*x[i])
      act= act + b

      if (y*act <= 0.0):
        for i in range(0,numvars):
          w[i] +=  (y*x[i])

        b = y + b
        
      else:
       correct+=1

      if(correct==rows):
        break

  #print(w,b)

  return (w,b)


# Compute the activation for input x.
# (NOTE: This should be a real-valued number, not simply +1/-1.)
def predict_perceptron(model, x):
  (w,b) = model
  correct = 0
  numvars = len(w)
  a = [0.0] * numvars
  sum = 0
  activation=0
  activation = 0.0

  for i in range(0,numvars):
    activation = activation + (w[i]*x[i])

  activation =activation+b

  return activation


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
  # Process command line arguments.
  # (You shouldn't need to change this.)
  if (len(argv) != 3):
    print('Usage: perceptron.py <train> <test> <model>')
    sys.exit(2)
  (train, varnames) = read_data(argv[0])
  (test, testvarnames) = read_data(argv[1])
  modelfile = argv[2]

  # Train model
  (w,b) = train_perceptron(train)
  #print ("Prediction begin")
  # Write model file
  # (You shouldn't need to change this.)
  f = open(modelfile, "w+")
  f.write('%f\n' % b)
  for i in range(len(w)):
    f.write('%s %f\n' % (varnames[i], w[i]))

  # Make predictions, compute accuracy
  correct = 0
  for (x,y) in test:
    activation = predict_perceptron( (w,b), x )
    #print(activation)
    if activation * y > 0:
      correct += 1
  acc = float(correct)/len(test)
  print("Accuracy: ",acc)

if __name__ == "__main__":
  main(sys.argv[1:])
