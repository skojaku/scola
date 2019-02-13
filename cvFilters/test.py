import numpy as np
from scipy import linalg
import pickle
import sys
import os
import pickle

from PMFG import PMFG 
from cvFilter import cvFilter
from LassoFilter import LassoFilter 
from ThresholdFilter import ThresholdFilter 
from cvNullModel import cvNullModel

def toCorr(C):			
	cov = np.asanyarray(C)
	std_ = np.sqrt(np.diag(cov))
	_C = cov / np.outer(std_, std_)
	return _C

def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)

#flist = [\
flist = ['Murayama-gs-im-sc',\
'JapanCov',\
'USCov',\
'cov105115_all',\
'cov118932_all']

sampleNum = {'Murayama-gs-im-sc':686, "cov105115_all":4760, "cov118932_all":4760, 'JapanCov':4930, 'USCov':5323}

for fname in flist:

	resfilename='res-%s.pickle' % fname
	lock='.lock-%s' % resfilename 
	resfilename = 'result/%s' % resfilename 
	lock = 'result/%s' % lock

	filename = "../data/null-config/null-config-%s.pickle" % fname
	
	with open(filename, 'rb') as f: 
		res = pickle.load(f)
	
	sCov = toCorr(res["Cov"])
	nCov = toCorr(res["corr"])
	
	sCov = np.matrix(sCov)	
	nCov = np.matrix(nCov)
	complexity = 2 * sCov.shape[0]
	
	# generate mask matrix
	N = nCov.shape[0]
	nm = cvNullModel()
	#nCov, complexity = nm.getNullCovariance(sCov, sampleNum[fname], 'eye')
	cvf = cvFilter()
	
	#nCov = np.matrix(np.eye(N))
	#mask = np.matrix(np.zeros((N, N)))
	#mask[0:5, :] = 1 
	#mask[:, 0:5] = 1
	#sCov = np.multiply( (1-mask), nCov) + np.multiply( mask, sCov)
	
	_filter_ = ThresholdFilter()
	_filter_ = PMFG()
	#_filter_ = LassoFilter()
	#G = _filter_.fit(sCov, nCov)
	#print(G)
	#_filter_ = LassoFilter()
	res = cvf.optimize_threshold(_filter_, sCov, nCov, sampleNum[fname], complexity)
	#print(res)
	
	#optimize_threshold(cvFilter = corFilter, sCov = sCov, nCov = nCov, sampleNum = sampleNum[fname])
	#model = modelSelection(filter=corFilter, alphalist = np.logspace(-2, 0, 100))
	#model = CovToNet_Thresholding()
	#print(model.dCov_)	
	sys.exit(0)
