import os
import numpy as np
import theano
import theano.tensor as T
from theano.compat.python2x import OrderedDict


class NADE(object):

    def __init__(self, random_seed, l_rate=None, input=None, input_size=400, hidden_size=200, tied=False):
        rng = np.random.RandomState(123)
        init_range = np.sqrt(6. / (hidden_size + input_size))

        # Set the number of visible units and hidden units in the network
        self.lr_rate = l_rate
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tied = tied
        self.input = input

        self.W = theano.shared(value=np.asarray(rng.uniform(low=-init_range, high=init_range, size=(input_size, hidden_size)), dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=np.zeros(hidden_size, dtype=theano.config.floatX), name='bhid', borrow=True)
        self.b_prime = theano.shared(value=np.zeros(input_size, dtype=theano.config.floatX), name='bvis', borrow=True)

        self.parameters = {str(self.W): self.W, str(self.b): self.b, str(self.b_prime): self.b_prime}

        # Init tied W
        if tied:
            self.W_prime = self.W
        else:
            self.W_prime = theano.shared(value=np.asarray(rng.uniform(low=-init_range, high=init_range, size=(input_size, hidden_size)), dtype=theano.config.floatX), name="W_prime", borrow=True)
            self.parameters[str(self.W_prime)] = self.W_prime

    def get_nll(self):
        input_times_W = self.input.T[:, :, None] * self.W[:, None, :]
        acc_input_times_W = T.concatenate([T.zeros_like(input_times_W[[0]]), T.cumsum(input_times_W, axis=0)[:-1]], axis=0)
        acc_input_times_W += self.b[None, None, :]
        h = T.nnet.sigmoid(acc_input_times_W)

        pre_output = T.sum(h * self.W_prime[:, None, :], axis=2) + self.b_prime[:, None]
        # output = T.nnet.sigmoid(pre_output)
        return T.sum(T.nnet.softplus(-self.input.T * pre_output + (1 - self.input.T) * pre_output), axis=0).mean()

    def get_cost_updates(self):
        mean_nll = self.get_nll()

        Wgrad = T.grad(mean_nll, self.W)
        bgrad = T.grad(mean_nll, self.b)
        b_primegrad = T.grad(mean_nll, self.b_prime)
        if not self.tied:
            w_primegrad = T.grad(mean_nll, self.W_prime)

        updates = OrderedDict()
        updates[self.W] = self.W - self.lr_rate * Wgrad
        updates[self.b] = self.b - self.lr_rate * bgrad
        updates[self.b_prime] = self.b_prime - self.lr_rate * b_primegrad
        if not self.tied:
            updates[self.W_prime] = self.W_prime - self.lr_rate * w_primegrad

        return (mean_nll, updates)

    def save(self, path):
        np.savez_compressed(os.path.join(path, "params"), self.parameters.values())
        # for param in self.params:
        #     np.save(path + str(param), param.get_value(borrow=True))

    def load(self, path):
        for param in np.load(os.path.join(path, "params.npz"))['arr_0']:
            self.parameters[str(param)].set_value(param.get_value())
