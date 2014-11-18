import numpy as np
import theano
import theano.tensor as T


class NADE(object):

    def __init__(self, random_seed, l_rate=None, input=None, n_visible=400, n_hidden=200, tied=False):

        rng = np.random.RandomState(123)
        init_range = np.sqrt(6. / (n_hidden + n_visible))

        # Set the number of visible units and hidden units in the network
        self.lr_rate = l_rate
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.tied = tied

        print "## initializing Model ##"
        self.W = theano.shared(value=np.asarray(rng.uniform(low=-init_range, high=init_range, size=(n_visible, n_hidden)), dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=np.zeros(n_hidden, dtype=theano.config.floatX), name='bhid', borrow=True)
        self.b_prime = theano.shared(value=np.zeros(n_visible, dtype=theano.config.floatX), name='bvis', borrow=True)

        self.params = [self.W, self.b, self.b_prime]

        # Init tied W
        if tied:
            self.W_prime = self.W
        else:
            self.W_prime = theano.shared(value=np.asarray(rng.uniform(low=-init_range, high=init_range, size=(n_visible, n_hidden)), dtype=theano.config.floatX), name="W_prime", borrow=True)
            self.params += [self.W_prime]

        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

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
