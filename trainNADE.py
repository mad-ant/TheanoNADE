from __future__ import division
import time as t
import numpy as np

import theano
import theano.tensor as T

import NADE


def get_done_text(start_time):
    return "DONE in {:.4f} seconds.".format(t.time() - start_time)


def trainNADE(src_folder, tgt_folder, batch_size=20, n_hid=40, learning_rate=0.1, training_epochs=40, gen_modelta=True, tied=False):

    print "## Loading dataset ...",
    start_time = t.time()
    dataset = np.load('binarized_mnist.npz')['arr_0'].item()
    print get_done_text(start_time)

    index = T.lscalar()
    x = T.matrix('x')

    print "## Initializing Model ...",
    start_time = t.time()
    model = NADE.NADE(random_seed=1234, l_rate=learning_rate, input=x, input_size=dataset['input_size'], n_hidden=n_hid, tied=tied)

    cost, updates = model.get_cost_updates()
    train_model = theano.function(inputs=[index],
                                  outputs=cost,
                                  updates=updates,
                                  givens={x: dataset['valid']['data'][index * batch_size:(index + 1) * batch_size]})

    test_model = theano.function(inputs=[index],
                                 outputs=cost,
                                 givens={x: dataset['test']['data'][index * batch_size:(index + 1) * batch_size]})
    print get_done_text(start_time)

    print "## Training batch={0} hidden_size={1} ##".format(batch_size, n_hid)
    start_time_train = t.time()
    for epoch in xrange(training_epochs):
        print "Epoch ", epoch

        nb_iterations = int(np.ceil(dataset['valid']['length'] / batch_size))
        print '\tTraining   ...',
        train_err = 0
        start_time = t.time()
        for index in range(nb_iterations):
            train_err += train_model(index)
        print get_done_text(start_time), " avg NLL: {0:.6f}".format(train_err / nb_iterations)

        nb_iterations_valid = int(np.ceil(dataset['valid']['length'] / batch_size))
        print '\tValidating ...',
        valid_err = 0
        start_time = t.time()
        for index in range(nb_iterations_valid):
            valid_err += test_model(index)
        print get_done_text(start_time), " NLL: {0:.6f}".format(valid_err / nb_iterations_valid)

    print "Complete training ", get_done_text(start_time_train)

    model.save_matrices(tgt_folder, "final")


trainNADE("data/bmnist/", "result/bmnist/", batch_size=100, n_hid=500, learning_rate=0.1, training_epochs=20, gen_modelta=True, tied=True)
