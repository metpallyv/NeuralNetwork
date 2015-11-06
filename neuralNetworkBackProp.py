__author__ = 'vardhaman'

import sys
import numpy as np
from numpy import genfromtxt

np.random.seed(1)

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

    #input weights of size no of inputs and no of hidden units
    wi = np.random.random(size=(iput.shape[0],no_hidden_units))
    #hidden layer weights of size of no of hidden units and size of inputs
    wo = np.random.random(size=(no_hidden_units,oput.shape[0]))
    #bias vector from input layer
    bias_input = np.ones(no_hidden_units)
    #bias vector from hidden layer
    bias_hidden = np.ones(iput.shape[0])
    #bias_hidden = bias_hidden.reshape(1,iput.shape[0])
    #run the algorithm till the termination condition is not satisfied for each instance

    for converge in xrange(10000):
        for inst in xrange(iput.shape[0]):
            #since we are running the algo for each example
            xi = iput[inst,:]
            xi = xi.reshape(1,iput.shape[0])
            target = xi

            #feedforward logic
            #activation function for hidden layer
            xh = sigmoid(np.dot(xi,wi) + bias_input)
            #activation function for output layer
            xo = sigmoid(np.dot(xh,wo)+ bias_hidden)

            #backpropogate error
            #error for output layer
            eOut = xo * (1 - xo) * (target - xo)
            eOut = eOut.reshape(1,iput.shape[0])

            #error for hidden layer
            ehidden = xh * (1 - xh) * np.dot(eOut,wo.T)
            ehidden = ehidden.reshape(1,no_hidden_units)

            #weight increment for input layer
            delta_wi = np.dot(xi.T,ehidden)
            delta_wi = learning_rate * delta_wi
            wi += delta_wi

            #weight increment for hidden layer
            delta_wo = np.dot(xh.T,eOut)
            delta_wo = learning_rate * delta_wo
            wo += delta_wo

            #update the bias for input layer
            for i in xrange(iput.shape[0]):
                for j in xrange(no_hidden_units):
                    bias_input[j] = bias_input[j]+(learning_rate * ehidden[0][j])

            #update the bias for hidden layer
            for j in xrange(no_hidden_units):
                for k in xrange(iput.shape[0]):
                    bias_hidden[j] = bias_hidden[j] = (learning_rate * eOut[0][k])

    #verifying if the ouput values would be same input
    for inst in xrange(iput.shape[0]):
        xi = iput[inst,:]
        xi = xi.reshape(1,iput.shape[0])
        xh = sigmoid(np.dot(xi,wi) + bias_input)
        print xh
        xo = sigmoid(np.dot(xh,wo)+ bias_hidden)
        for i, x in enumerate(xo):
            Val = map(lambda x: 1 if x > 0.5 else 0,x)
            print Val

#main code
if __name__ == '__main__':
    if len(sys.argv) == 4:
        training_file = sys.argv[1]
        no_hidden_units = int(sys.argv[2])
        learning_rate = float(sys.argv[3])
        input_arr,output_arr = load_file(training_file)
        train_back_prop(input_arr,output_arr,no_hidden_units,learning_rate)
