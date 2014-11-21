from __future__ import division
import os
import sys
import time as t
import numpy as np

import theano
import theano.tensor as T

import NADE


def get_done_text(start_time):
    return "DONE in {:.4f} seconds.".format(t.time() - start_time)


def trainNADE(save_path, batch_size=100, hidden_size=500, learning_rate=0.05, max_epochs=100, look_ahead=15, tied=False):
    save_path = "{}/{}_{}_{}_{}_{}_{}".format(save_path, batch_size, hidden_size, learning_rate, max_epochs, look_ahead, tied)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print "## Loading dataset ...",
    start_time = t.time()
    dataset = np.load('binarized_mnist.npz')['arr_0'].item()
    print get_done_text(start_time)

    index = T.lscalar()
    input = T.matrix('input')

    print "## Initializing Model ...",
    start_time = t.time()
    model = NADE.NADE(random_seed=1234, l_rate=learning_rate, input=input, input_size=dataset['input_size'], hidden_size=hidden_size, tied=tied)

    cost, updates = model.get_cost_updates()
    train_model = theano.function(inputs=[index],
                                  outputs=cost,
                                  updates=updates,
                                  givens={input: dataset['train']['data'][index * batch_size:(index + 1) * batch_size]})

    train_nll = theano.function(inputs=[index],
                                outputs=cost,
                                givens={input: dataset['train']['data'][index * batch_size:(index + 1) * batch_size]})

    valid_nll = theano.function(inputs=[index],
                                outputs=cost,
                                givens={input: dataset['valid']['data'][index * batch_size:(index + 1) * batch_size]})

    test_nll = theano.function(inputs=[index],
                               outputs=cost,
                               givens={input: dataset['test']['data'][index * batch_size:(index + 1) * batch_size]})
    #theano.printing.pydotprint(train_model, 'train_model')
    print get_done_text(start_time)

    print "## Training NADE: batch={0} hidden_size={1} tied={2} ##".format(batch_size, hidden_size, tied)
    start_time_train = t.time()

    trainer_status = {
        "best_valid_error": np.inf,
        "best_epoch": 0,
        "epoch": 0,
        "nb_of_epocs_without_improvement": 0
    }

    while(trainer_status["epoch"] < max_epochs and trainer_status["nb_of_epocs_without_improvement"] < look_ahead):
        trainer_status["epoch"] += 1

        print "Epoch ", trainer_status["epoch"]
        print '\tTraining   ...',
        train_err = 0
        nb_iterations = int(np.ceil(dataset['train']['length'] / batch_size))
        start_time = t.time()
        for index in range(nb_iterations):
            train_err += train_model(index)
        print get_done_text(start_time), " avg NLL: {0:.6f}".format(train_err / nb_iterations)

        nb_iterations_valid = int(np.ceil(dataset['valid']['length'] / batch_size))
        print '\tValidating ...',
        valid_err = 0
        start_time = t.time()
        for index in range(nb_iterations_valid):
            valid_err += valid_nll(index)
        print get_done_text(start_time), " NLL: {0:.6f}".format(valid_err / nb_iterations_valid)

        if valid_err < trainer_status["best_valid_error"]:
            trainer_status["best_valid_error"] = valid_err
            trainer_status["best_epoch"] = trainer_status["epoch"]
            trainer_status["nb_of_epocs_without_improvement"] = 0
            model.save(save_path)
        else:
            trainer_status["nb_of_epocs_without_improvement"] += 1

    print "### Training", get_done_text(start_time_train), "###"

    print '\n### Evaluating best model from Epoch {0} ###'.format(trainer_status["best_epoch"])
    model.load(save_path)
    for subset in ['test', 'valid', 'train']:
        nll = locals()["{}_nll".format(subset)]
        err = 0
        nb_iterations = int(np.ceil(dataset[subset]['length'] / batch_size))
        for index in range(nb_iterations):
            err += nll(index)
        print "\tBest {1} error is : {0:.6f}".format(err / nb_iterations, subset)


def parse_args(args):
    import argparse

    class GroupedAction(argparse.Action):

        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            super(GroupedAction, self).__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            group = self.container.title
            dest = self.dest
            groupspace = getattr(namespace, group, argparse.Namespace())
            setattr(groupspace, dest, values)
            setattr(namespace, group, groupspace)

    parser = argparse.ArgumentParser(description='Train the NADE model.')

    group_trainer = parser.add_argument_group('train')
    group_trainer.add_argument('learning_rate', type=float, action=GroupedAction, default=argparse.SUPPRESS)
    group_trainer.add_argument('max_epochs', type=lambda x: np.inf if x == "-1" else int(x), help="If -1 will run until convergence.", action=GroupedAction, default=argparse.SUPPRESS)
    group_trainer.add_argument('batch_size', type=int, action=GroupedAction, default=argparse.SUPPRESS)
    group_trainer.add_argument('look_ahead', type=int, action=GroupedAction, default=argparse.SUPPRESS)

    group_model = parser.add_argument_group('model')
    group_model.add_argument('hidden_size', type=int, action=GroupedAction, default=argparse.SUPPRESS)
    #group_model.add_argument('random_seed', type=int, action=GroupedAction, default=argparse.SUPPRESS)
    group_model.add_argument('tied', type=bool, action=GroupedAction, default=argparse.SUPPRESS)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args(sys.argv)
    hyperparams = vars(args.model)
    trainingparams = vars(args.train)

    trainNADE("result/bmnist/", tied=hyperparams['tied'], hidden_size=hyperparams['hidden_size'], look_ahead=trainingparams['look_ahead'], batch_size=trainingparams['batch_size'], learning_rate=trainingparams['learning_rate'], max_epochs=trainingparams['max_epochs'])
