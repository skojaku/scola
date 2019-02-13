import numpy as np

class MLEFilter():
	
	def __init__(self):
		pass
	
	def fit(self, sCov, nCov, dCov, sampleNum, alpha=None):
		N = sCov.shape[0] 
		dCov = sCov - nCov
		return dCov
 
	def loglikelihood(self, dCov, sCov, nCov):
		Cov = dCov + nCov
		s, v = np.linalg.slogdet(Cov)
		return -0.5 *  s * v -0.5*np.trace(sCov @ linalg.inv(Cov)) - 0.5 * Cov.shape[0] * np.log(2 * np.pi)

	def candidateThreshold(self, sCov, nCov, num=30):
		return [1]
