from __future__ import division
import time
import numpy as np

import theano
import theano.tensor as T

import NADE


def trainNADE(src_folder, tgt_folder, batch_size=20, n_hid=40, learning_rate=0.1, training_epochs=40, gen_data=True, tied=False):

    print "## Loading dataset ...",
    dataset = np.load('binarized_mnist.npz')['arr_0'].item()
    print "Done"

    index = T.lscalar()
    x = T.matrix('x')

    print "## Initializing Model ...",
    start_time = time.clock()
    da = NADE.NADE(random_seed=1234, l_rate=learning_rate, input=x, n_visible=dataset['input_size'], n_hidden=n_hid, tied=tied)

    cost, updates = da.get_cost_updates()
    train_da = theano.function(inputs=[index],
                               outputs=cost,
                               updates=updates,
                               givens={x: dataset['train']['data'][index * batch_size:(index + 1) * batch_size]})

    vcost = da.get_cost()
    test_da = theano.function(inputs=[],
                              outputs=vcost,
                              givens={x: dataset['test']['data']})
    print "Done\n"

    print "## Training batch={0} ##".format(batch_size)
    for epoch in xrange(training_epochs):
        print "Epoch ", epoch

        nb_iterations = int(np.ceil(dataset['train']['length'] / batch_size))
        print '\tTraining   ...',
        train_err = 0
        for index in range(nb_iterations):
            print index
            train_err += train_da(index)
        print "Done. avg NLL: {0:.6f}".format(train_err / nb_iterations)

        nb_iterations_valid = int(np.ceil(dataset['valid']['length'] / batch_size))
        print '\tValidating ...',
        for index in range(nb_iterations_valid):
            valid_err = test_da(index)
        print "Done. NLL: {0:.6f}".format(valid_err / nb_iterations_valid)

    training_time = time.clock() - start_time

    print ' code ran for %.2fm' % (training_time / 60.)
    da.save_matrices(tgt_folder, "final")


trainNADE("data/bmnist/", "result/bmnist/", batch_size=100, n_hid=500, learning_rate=0.1, training_epochs=20, gen_data=True, tied=True)
