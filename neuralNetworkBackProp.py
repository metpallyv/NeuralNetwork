__author__ = 'vardhaman'

import sys
import numpy as np
import random
from numpy import genfromtxt
import math

np.random.seed(0)

#load the input file to be auto encoded. Since the output should be same as i/p, return the same np array as o/p
def load_file(file):
     Y = genfromtxt(file, delimiter=",",dtype=str)
     X = Y.astype(np.float)
     return (X,X)

#sigmoid function
def sigmoid(X):
    return 1 / (1 + np.exp(- X))

#this function is used to train and backprop algm for the given input
def train_back_prop(iput,oput,no_hidden_units,learning_rate):

    wi = np.random.random(size=(iput.shape[0],no_hidden_units))
    wo = np.random.random(size=(no_hidden_units,oput.shape[0]))
    bias_input = np.ones(no_hidden_units)
    bias_hidden = np.ones(iput.shape[0])
    #bias_hidden = bias_hidden.reshape(1,iput.shape[0])
    #run the algorithm till the termination condition is not satisfied for each instance

    for converge in xrange(1000):
        for inst in xrange(iput.shape[0]):
            xi = iput[inst,:]
            xi = xi.reshape(1,iput.shape[0])
            target = xi
            xh = sigmoid(np.dot(xi,wi) + bias_input)
            xo = sigmoid(np.dot(xh,wo)+ bias_hidden)
            
            eOut = xo * (1 - xo) * (target - xo)
            eOut = eOut.reshape(1,iput.shape[0])
            #print eOut

            ehidden = xh * (1 - xh) * np.dot(eOut,wo.T)
            ehidden = ehidden.reshape(1,no_hidden_units)

            delta_wi = np.dot(xi.T,ehidden)
            delta_wi = learning_rate * delta_wi
            wi += delta_wi
            
            delta_wo = np.dot(xh.T,eOut)
            delta_wo = learning_rate * delta_wo
            wo += delta_wo

            for i in xrange(iput.shape[0]):
                for j in xrange(no_hidden_units):
                    bias_input[j] = bias_input[j]+(learning_rate * ehidden[0][j])

            for j in xrange(no_hidden_units):
                for k in xrange(iput.shape[0]):
                    bias_hidden[j] = bias_hidden[j] = (learning_rate * eOut[0][k])
                    
    for inst in xrange(iput.shape[0]):
        xi = iput[inst,:]
        xi = xi.reshape(1,iput.shape[0])
        xh = sigmoid(np.dot(xi,wi) + bias_input)
        xo = sigmoid(np.dot(xh,wo)+ bias_hidden)
        for i, x in enumerate(xo):
            Val = map(lambda x: 1 if x > 0.5 else 0,x)
            print Val

if __name__ == '__main__':
    if len(sys.argv) == 4:
        training_file = sys.argv[1]
        no_hidden_units = int(sys.argv[2])
        learning_rate = float(sys.argv[3])
        input_arr,output_arr = load_file(training_file)
        train_back_prop(input_arr,output_arr,no_hidden_units,learning_rate)
