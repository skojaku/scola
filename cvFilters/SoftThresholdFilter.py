import numpy as np
from scipy import linalg

class SoftThresholdFilter():
	
	def __init__(self):
		pass
	
	def fit(self, sCov, nCov, dCov, sampleNum, alpha):
		N = sCov.shape[0] 
		return self._prox(sCov - nCov, alpha)

	def _prox(self, y, alpha):
		return np.multiply(np.sign(y),  np.maximum(np.abs(y) - alpha, np.matrix(np.zeros(y.shape))))
 
	def _fast_inv_mat_lapack(self, M):
		zz , _ = linalg.lapack.dpotrf(M, False, False)
		inv_M , info = linalg.lapack.dpotri(zz)
		inv_M = np.triu(inv_M) + np.triu(inv_M, k=1).T
		return inv_M
	def loglikelihood(self, dCov, sCov, nCov):
		Cov = dCov + nCov
		try:
			w, v = np.linalg.eig(Cov)
			if np.min(w) < 0:
				return -1e+30
			iCov = self._fast_inv_mat_lapack(Cov)
			return -0.5 *  np.sum(np.log(w))  - 0.5*np.trace(sCov @ iCov) - 0.5 * Cov.shape[0] * np.log(2 * np.pi)
		except:
				return -1e+30
		return -0.5 *  s * v -0.5*np.trace(sCov @ linalg.inv(Cov)) - 0.5 * Cov.shape[0] * np.log(2 * np.pi)

	def candidateThreshold(self, sCov, nCov, num=30):
		return np.logspace(-10, 0, num)[::-1]	

	def ind2sub(self,array_shape, ind):
		rows = np.floor(ind.astype('int') / array_shape[1]).astype(int)
		cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
		return (rows, cols)
