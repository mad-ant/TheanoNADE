import numpy
from os import listdir
from os.path import isfile, join


def denseloader(file, bit):
    print "loading ...", file
    mat = numpy.load(file + ".npy")
    mat = numpy.array(mat, dtype=bit)
    return mat


def sparseloader(filename, bit, row, col):
    print "loading ...", filename
    x = numpy.load(filename + "d.npy")
    y = numpy.load(filename + "i.npy")
    z = numpy.load(filename + "p.npy")
    mat = sparse.csr_matrix((x, y, z), shape=(row, col))
    mat = mat.todense()
    return mat


def prepare_data(src_folder, tgt_folder, fts, batch_size):

    fcount = 0

    ipfile = open(tgt_folder + "ip.txt", "w")

    trainfiles = [src_folder + f for f in listdir(src_folder) if isfile(join(src_folder, f))]

    for filename in trainfiles:
        file = open(filename, "r")

        mat = numpy.zeros((1000, fts))
        count = 0

        for line in file:
            line = line.strip().split()
            for i in range(0, fts):
                mat[count][i] = float(line[i])
            count += 1
            if(count == 1000):
                numpy.save(tgt_folder + str(fcount), mat)
                ipfile.write(tgt_folder + str(fcount) + "," + str(1000 / batch_size) + "\n")
                mat = numpy.zeros((1000, fts))
                count = 0
                fcount += 1
        file.close()

    ipfile.close()
