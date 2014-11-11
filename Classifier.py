import sys
import numpy as np
import numpy

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.cross_decomposition import CCA


def svm_classifier(trainx, trainy, testx, testy):

	x = numpy.load(trainx)
	x = x[:,0:50]
	y = numpy.load(trainy)
	clf = svm.LinearSVC()
	clf.fit(x,y)

	xx = numpy.load(testx)
	xx = xx[:,50:100]
	yy = numpy.load(testy)
	pred = clf.predict(xx)
	return accuracy_score(yy,pred)



def cca(trainx, testx, ntrainx, ntestx):

	x = numpy.load(trainx)

	x1 = x[:,0:392]
	x2 = x[:,392:784]

	cc = CCA(n_components = 50)

	cc.fit(x1,x2)

	print "fitted"

	m,n = cc.transform(x1,x2)

	newx = numpy.concatenate((m,n),axis = 1)

	numpy.save(ntrainx,newx)

	x = numpy.load(testx)

	x1 = x[:,0:392]
	x2 = x[:,392:784]

	m,n = cc.transform(x1,x2)

	newx = numpy.concatenate((m,n),axis = 1)

	numpy.save(ntestx,newx)



print svm_classifier(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])

#cca(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])