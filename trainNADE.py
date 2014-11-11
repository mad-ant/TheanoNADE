from os import listdir
from os.path import isfile, join

import os
import time
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import NNUtil
import NADE


def load(file, train_set_x, nvis):

    mat = NNUtil.denseloader(file, "float32")
    train_set_x.set_value(mat, borrow=True)


def trainNADE(src_folder, tgt_folder, batch_size=20, n_hid=40, learning_rate=0.1, training_epochs=40, gen_data=True, tied=False):

    mat_pic = src_folder + "mat_pic/"

    if not os.path.exists(mat_pic):
        os.makedirs(mat_pic)

    if not os.path.exists(mat_pic + "train/"):
        os.makedirs(mat_pic + "train/")

    if not os.path.exists(mat_pic + "test/"):
        os.makedirs(mat_pic + "test/")

    if not os.path.exists(mat_pic + "valid/"):
        os.makedirs(mat_pic + "valid/")

    trainfiles = [src_folder + "train/" + f for f in listdir(src_folder + "train/") if isfile(join(src_folder + "train/", f))]

    file = open(trainfiles[0], "r")
    n_vis = len(file.readline().strip().split())
    file.close()

    if gen_data:
        NNUtil.prepare_data(src_folder + "train/", src_folder + "mat_pic/train/", n_vis, batch_size)
        NNUtil.prepare_data(src_folder + "valid/", src_folder + "mat_pic/valid/", n_vis, batch_size)
        NNUtil.prepare_data(src_folder + "test/", src_folder + "mat_pic/test/", n_vis, batch_size)

    index = T.lscalar()
    x = T.matrix('x')
    l_rate = T.fscalar()

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = NADE.NADE(numpy_rng=rng, theano_rng=theano_rng, l_rate=l_rate, input=x, n_visible=n_vis, n_hidden=n_hid, tied=tied)

    start_time = time.clock()

    train_set_x = theano.shared(np.asarray(np.zeros((1000, n_vis)), dtype=theano.config.floatX), borrow=True)
    shared_l_rate = theano.shared(np.asarray(0.1, dtype='float32'), borrow=True)

    cost, updates = da.get_cost_updates()
    train_da = theano.function([index], cost, updates=updates, givens=[(x, train_set_x[index * batch_size:(index + 1) * batch_size]), (l_rate, shared_l_rate)])

    vcost = da.get_cost()
    test_da = theano.function([index], vcost, givens=[(x, train_set_x[index * batch_size:(index + 1) * batch_size])])

    diff = 0
    flag = 1

    detfile = open(tgt_folder + "details.txt", "w")
    detfile.close()

    oldtc = float("inf")

    for epoch in xrange(training_epochs):

        print "in epoch ", epoch

        c = []

        ipfile = open(src_folder + "mat_pic/train/ip.txt", "r")

        for line in ipfile:
            next = line.strip().split(",")
            load(next[0], train_set_x, n_vis)
            for batch_index in range(0, int(next[1])):
                if(batch_index % 100 == 0):
                    print batch_index
                c.append(train_da(batch_index))

        if(flag == 1):
            flag = 0
            diff = np.mean(c)
            di = diff
        else:
            di = np.mean(c) - diff
            diff = np.mean(c)

            print 'Difference between 2 epochs is ', di
        print 'Training epoch %d, cost ' % epoch, diff

        ipfile.close()

        detfile = open(tgt_folder + "details.txt", "a")
        detfile.write(str(diff) + "\n")
        detfile.close()
        # save the parameters for every 5 epochs
        if((epoch + 1) % 5 == 0):
            da.save_matrices(tgt_folder, str(epoch))

        print "validating"

        tc = []
        ipfile = open(src_folder + "mat_pic/valid/ip.txt", "r")

        for line in ipfile:
            next = line.strip().split(",")
            load(next[0], train_set_x, n_vis)
            for batch_index in range(0, int(next[1])):
                if(batch_index % 100 == 0):
                    print batch_index
                tc.append(test_da(batch_index))

        cur_tc = np.mean(tc)

        print cur_tc
        if(cur_tc < oldtc):
            oldtc = cur_tc
        else:
            oldtc = cur_tc
            m = shared_l_rate.get_value(borrow=True) * 0.5
            shared_l_rate.set_value(m, borrow=True)
            print "updated lrate"

    end_time = time.clock()

    training_time = (end_time - start_time)

    print ' code ran for %.2fm' % (training_time / 60.)
    da.save_matrices(tgt_folder, "final")


trainNADE("../data/bmnist/", "../result/bmnist/", batch_size=20, n_hid=500, learning_rate=0.1, training_epochs=20, gen_data=True, tied=True)
