import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


class NADE(object):

    def __init__(self, numpy_rng, l_rate=None, theano_rng=None, input=None, n_visible=400, n_hidden=200, W=None, bhid=None, bvis=None, W_prime=None, tied=False):

        # Set the number of visible units and hidden units in the network
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.tied = tied

        # Random seed
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not W:
            print "randomly initializing W"
            initial_W = np.asarray(numpy_rng.uniform(low=-1 * np.sqrt(6. / (n_hidden + n_visible)), high=1 * np.sqrt(6. / (n_hidden + n_visible)), size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            initial_W = initial_W * 4
            W = theano.shared(value=initial_W, name='W', borrow=True)

        else:
            print "loading W matrix"
            initial_W = np.load(W + ".npy")
            W = theano.shared(value=initial_W, name='W', borrow=True)

        self.W = W

        if not tied:

            if not W_prime:
                print "randomly initializing W_prime"
                initial_W_prime = np.asarray(numpy_rng.uniform(low=-4 * np.sqrt(6. / (n_hidden + n_visible)), high=4 * np.sqrt(6. / (n_hidden + n_visible)), size=(n_visible, n_hidden)), dtype=theano.config.floatX)
                W_prime = theano.shared(value=initial_W_prime, name='W_prime', borrow=True)
            else:
                print "loading W_prime matrix"
                initial_W_prime = np.load(W_prime + ".npy")
                W_prime = theano.shared(value=initial_W_prime, name="W_prime", borrow=True)

            self.W_prime = W_prime

        else:
            self.W_prime = self.W

        if not bhid:
            print "randomly initializing hidden bias"
            bhid = theano.shared(value=np.zeros(n_hidden, dtype=theano.config.floatX), name='bhid', borrow=True)

        else:
            print "loading hidden bias"
            initial_bhid = np.load(bhid + ".npy")
            bhid = theano.shared(value=initial_bhid, name='bhid', borrow=True)

        self.b = bhid

        if not bvis:
            print "randomly initializing visible bias"
            bvis = theano.shared(value=np.zeros(n_visible, dtype=theano.config.floatX), borrow=True)

        else:
            print "loading visible bias"
            initial_bvis = np.load(bvis + ".npy")
            bvis = theano.shared(value=initial_bvis, name='bvis', borrow=True)

        self.b_prime = bvis

        self.theano_rng = theano_rng

        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.lr_rate = l_rate

        if tied:
            self.params = [self.W, self.b, self.b_prime]
        else:
            self.params = [self.W, self.b, self.b_prime, self.W_prime]

    def get_nll(self):

        def cum(W, X, V, bp, p_prev, a_prev, x_prev, b):

            a = a_prev + theano.dot(T.shape_padright(x_prev, 1), T.shape_padleft(W, 1))
            h = T.nnet.sigmoid(a + b)
            o = T.nnet.sigmoid(T.dot(h, V) + bp)
            p = p_prev - (X * T.log(o) + (1 - X) * T.log(1 - o))
            return p, a, X

        ([res1, _, _], updates) = theano.scan(fn=cum, outputs_info=[T.zeros_like(self.x.T[0]), T.zeros_like(theano.dot(self.x, self.W)), T.zeros_like(self.x.T[0])], sequences=[self.W, self.x.T, self.W_prime, self.b_prime], non_sequences=self.b)

        return (res1[-1], updates)

    def get_cost_updates(self):

        nll, updates = self.get_nll()
        mean_nll = nll.mean()

        Wgrad = T.grad(mean_nll, self.W)
        bgrad = T.grad(mean_nll, self.b)
        b_primegrad = T.grad(mean_nll, self.b_prime)

        if not self.tied:
            w_primegrad = T.grad(mean_nll, self.W_prime)

        updates[self.W] = self.W - self.lr_rate * Wgrad
        updates[self.b] = self.b - self.lr_rate * bgrad
        updates[self.b_prime] = self.b_prime - self.lr_rate * b_primegrad

        if not self.tied:
            updates[self.W_prime] = self.W_prime - self.lr_rate * w_primegrad

        return (mean_nll, updates)

    def get_cost(self):

        nll, updates = self.get_nll()
        mean_nll = nll.mean()

        return mean_nll

    # This method saves W, bvis and bhid matrices. `n` is the string attached to the file name.
    def save_matrices(self, folder, n):

        np.save(folder + "b" + n, self.b.get_value(borrow=True))
        np.save(folder + "bp" + n, self.b_prime.get_value(borrow=True))
        np.save(folder + "w" + n, self.W.get_value(borrow=True))

        if not self.tied:
            np.save(folder + "w_prime" + n, self.W_prime.get_value(borrow=True))
