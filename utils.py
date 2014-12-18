import sys
import time as t
import pickle
import json
import theano
import hashlib
import fcntl
#from GoogleSheetAPI import GSheet#
import os
from contextlib import contextmanager
import logging


@contextmanager
def open_with_lock(*args, **kwargs):
    """ Context manager for opening file with an exclusive lock. """
    f = open(*args, **kwargs)
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        logging.info("Can't immediately write-lock the file ({0}), blocking ...".format(f.name))
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    yield f
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    f.close()


def generate_uid_from_string(value):
    """ Create unique identifier from a string. """
    return hashlib.sha256(value).hexdigest()


def saveFeatures(model):
    features = model.layers[0].W.get_value()
    # for layer in model.layers[1:]:
    #     w = layer.W.get_value()
    #     print "we", w.shape, features.shape
    #     if w.shape[1] != features.shape[1]:
    #         print "transpose"
    #         w = w.T
    #     features = np.c_[features, w]
    pickle.dump(features, open('W.pkl', 'wb'))


def write_result(dataset_name, model_info, experiment_name):
    result_file = "_RESULTS_{0}_nadeTheano.csv".format(dataset_name)
    write_header = not os.path.exists(result_file)

    header = ["Learning Rate", "Decrease Constant", "Hidden Layers", "Random Seed", "Activation Function", "Tied Weights", "Max Epoch", "Best Epoch", "Look Ahead", "Batch Size", "Momentum", "Dropout Rate", "Weights Initialization", "Training err", "Training err std", "Validation err", "Validation err std", "Test err", "Test err std", "Total Training Time", "Experiment Name"]

    with open_with_lock(result_file, "a") as f:
        if write_header:
            f.write("\t".join(header) + "\n")
        f.write('\t'.join(map(str, model_info[0:13])) + "\t" + '\t'.join(map("{0:.6f}".format, model_info[13:-1])) + "\t" + "{0:.4f}".format(model_info[-1]) + "\t{0}\n".format(experiment_name))


# def write_result(dataset_name, model_info, experiment_name):
# print "### Savin result to the cloud ^^ ...",
#     start_time = t.time()

#     header = ["Learning Rate", "Decrease Constant", "Hidden Layers", "Random Seed", "Activation Function", "Tied Weights", "Max Epoch", "Best Epoch", "Look Ahead", "Batch Size", "Momentum", "Dropout Rate", "Weights Initialization", "Training err", "Training err std", "Validation err", "Validation err std", "Test err", "Test err std", "Total Training Time", "Experiment Name"]

#     sheet = GSheet('1gwMuCMHE0mhu7EMP74djEYARfSvnBrMnsAE-cTGJS2w', 'mathieu.germain@gmail.com', 'nigvtrjsrshqzsqb')
#     worksheetID = sheet.getWorksheetID(dataset_name)

#     if worksheetID is None:
#         worksheetID = sheet.createWorksheet(dataset_name, header)

#     result = model_info[0:21]
#     result += map("{0:.6f}".format, model_info[21:-1])
#     result += ["{0:.4f}".format(model_info[-1])]
#     result += [experiment_name]

#     try:
#         sheet.addRow(worksheetID, result)
#     except:
#         sheet.addRow(worksheetID, result)

#     print get_done_text(start_time)


def get_done_text(start_time):
    sys.stdout.flush()
    return "DONE in {:.4f} seconds.".format(t.time() - start_time)


def save_dict_to_json_file(path, dictionary):
    with open(path, "w") as json_file:
        json_file.write(json.dumps(dictionary, indent=4, separators=(',', ': ')))


def load_dict_from_json_file(path):
    with open(path, "r") as json_file:
        return json.loads(json_file.read())


def print_computational_graphs(model, hidden_sizes, shuffle_mask, use_cond_mask, direct_input_connect):
    file_name = "MADE_h{}.shuffle{}.cond_mask{}.direct_conn{}_{}_".format(hidden_sizes, shuffle_mask, use_cond_mask, direct_input_connect, theano.config.device)
    theano.printing.pydotprint(model.use, file_name + "use")
    theano.printing.pydotprint(model.learn, file_name + "learn")
    theano.printing.pydotprint(model.shuffle, file_name + "shuffle")
    theano.printing.pydotprint(model.valid_log_prob, file_name + "log_prob")
    exit()
