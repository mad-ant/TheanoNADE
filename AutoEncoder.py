# Author : Sarath Chandar


import os
import sys
import time
from numpy import *
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from scipy import sparse





class Autoencoder(object):

    def __init__(self, numpy_rng, l_rate=None, theano_rng=None, input=None,n_visible=400, n_hidden=200,W=None, bhid=None, bvis=None, W_prime = None, tied = False, hidden_activation = "sigmoid", output_activation = "identity", loss = "squarrederror"):


        # Set the number of visible units and hidden units in the network
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.tied = tied
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss

        # Random seed
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not W:
            print "randomly initializing W"
            initial_W = numpy.asarray(numpy_rng.uniform(low=-1 * numpy.sqrt(6. / (n_hidden + n_visible)),high=1 * numpy.sqrt(6. / (n_hidden + n_visible)),size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            if self.hidden_activation == "sigmoid":
                initial_W = initial_W * 4
            W = theano.shared(value=initial_W, name='W', borrow=True)

        else:
            print "loading W matrix"
            initial_W = numpy.load(W+".npy")
            W = theano.shared(value=initial_W, name='W', borrow=True)

        self.W = W

        if tied==False:

            if not W_prime:
                print "randomly initializing W_prime"
                initial_W_prime = numpy.asarray(numpy_rng.uniform(low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),size=(n_hidden, n_visible)), dtype=theano.config.floatX)
                W_prime = theano.shared(value=initial_W_prime, name='W_prime', borrow=True)
            else:
                print "loading W_prime matrix"
                initial_W_prime = numpy.load(W_prime+".npy")
                W_prime = theano.shared(value = initial_W_prime, name = "W_prime", borrow = True)

            self.W_prime = W_prime

        else:
            self.W_prime = self.W.T

        if not bhid:
            print "randomly initializing hidden bias"
            bhid = theano.shared(value=numpy.zeros(n_hidden,dtype=theano.config.floatX),name='bhid',borrow=True)

        else:
            print "loading hidden bias"
            initial_bhid = numpy.load(bhid+".npy")
            bhid = theano.shared(value=initial_bhid, name='bhid', borrow=True)

        self.b = bhid


        if not bvis:
            print "randomly initializing visible bias"
            bvis = theano.shared(value=numpy.zeros(n_visible,dtype=theano.config.floatX),borrow=True)

        else:
            print "loading visible bias"
            initial_bvis = numpy.load(bvis+".npy")
            bvis = theano.shared(value=initial_bvis, name='bvis', borrow=True)


        self.b_prime = bvis



        self.theano_rng = theano_rng

        if input == None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.lr_rate = l_rate


        if(tied==True):
            self.params = [self.W,  self.b, self.b_prime]
        else:
            self.params = [self.W, self.b, self.b_prime, self.W_prime]



    def get_cost_updates(self):


        if(self.hidden_activation=="sigmoid"):
            y = T.nnet.sigmoid(T.dot(self.x, self.W) + self.b)
        elif(self.hidden_activation=="tanh"):
            y = T.tanh(T.dot(self.x, self.W) + self.b)

        if(self.output_activation == "sigmoid"):
            z = T.nnet.sigmoid(T.dot(y, self.W_prime) + self.b_prime)
        elif(self.output_activation == "identity"):
            z = (T.dot(y, self.W_prime) + self.b_prime)


        if(self.loss == "squarrederror"):
            L = T.sum(T.sqr(self.x-z)/2,axis = 1)
        elif(self.loss == "crossentrophy"):
            L = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)

        cost = T.mean(L)


        Wgrad = T.grad(cost, self.W)
        bgrad = T.grad(cost, self.b)
        b_primegrad = T.grad(cost, self.b_prime)

        if(self.tied==False):
            w_primegrad = T.grad(cost, self.W_prime)


        updates = []

        updates.append((self.W, self.W - self.lr_rate * Wgrad))
        updates.append((self.b, self.b - self.lr_rate * bgrad))
        updates.append((self.b_prime, self.b_prime - self.lr_rate * b_primegrad))

        if(self.tied == False):
            updates.append((self.W_prime, self.W_prime - self.lr_rate * w_primegrad))

        return (cost, updates)


    def get_cost(self):


        if(self.hidden_activation=="sigmoid"):
            y = T.nnet.sigmoid(T.dot(self.x, self.W) + self.b)
        elif(self.hidden_activation=="tanh"):
            y = T.tanh(T.dot(self.x, self.W) + self.b)

        if(self.output_activation == "sigmoid"):
            z = T.nnet.sigmoid(T.dot(y, self.W_prime) + self.b_prime)
        elif(self.output_activation == "identity"):
            z = (T.dot(y, self.W_prime) + self.b_prime)


        if(self.loss == "squarrederror"):
            L = T.sum(T.sqr(self.x-z)/2,axis = 1)
        elif(self.loss == "crossentrophy"):
            L = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)

        cost = T.mean(L)



        return cost



    # This method saves W, bvis and bhid matrices. `n` is the string attached to the file name.
    def save_matrices(self,folder,n):

        numpy.save(folder+"b"+n, self.b.get_value(borrow=True))
        numpy.save(folder+"bp"+n, self.b_prime.get_value(borrow=True))
        numpy.save(folder+"w"+n, self.W.get_value(borrow=True))

        if(self.tied == False):
            numpy.save(folder+"w_prime"+n,self.W_prime.get_value(borrow=True))

