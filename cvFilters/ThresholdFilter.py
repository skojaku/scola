import numpy as np
from scipy import linalg

class ThresholdFilter():
	
	def __init__(self):
		pass
	
	def fit(self, sCov, nCov, dCov, sampleNum, alpha):
		N = sCov.shape[0] 
		dCov = sCov - nCov
		
		v = np.array(np.abs(dCov)).ravel()
		idx = np.argsort(v)
		m = np.round((1-alpha) * N * N).astype(int) 
		r,c = self.ind2sub((N, N), idx[0:m])
		dCov[r,c] = 0
		return dCov
 
	def loglikelihood(self, dCov, sCov, nCov):
		Cov = dCov + nCov
		s, v = np.linalg.slogdet(Cov)
		return -0.5 *  s * v -0.5*np.trace(sCov @ linalg.inv(Cov)) - 0.5 * Cov.shape[0] * np.log(2 * np.pi)

	def candidateThreshold(self, sCov, nCov, num=30):
		return [0.001, 0.01, 0.1, 0.9]

	def ind2sub(self,array_shape, ind):
		rows = np.floor(ind.astype('int') / array_shape[1]).astype(int)
		cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
		return (rows, cols)
