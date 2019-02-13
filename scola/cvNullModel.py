import numpy as np
import configcorr as cg

class cvNullModel():
		
	def __init__(self):
		pass
	
	def getNullCovariance(self, sCov, sampleNum, null_model):
		
		if null_model == "eye":	
			return np.matrix(np.eye(sCov.shape[0])), 0 
		elif null_model == "mg2":	
			return self._MG2(sCov, sampleNum)
		elif null_model == "mg3":	
			return self._MG3(sCov, sampleNum)
		elif null_model == "hqs":	
			mu_on = np.mean(np.diagonal(sCov)) 
			mu_off = np.mean(np.triu(sCov,1))
			v = np.std(np.triu(sCov,1))
			nCov = mu_off * np.ones(sCov.shape)
			np.fill_diagonal(nCov, 1) 
			return nCov, 1
		elif null_model == "config":
			# not implemented yet
			nCov = cg.max_ent_config_dmcc(sCov, 1e-4, True)
			std_ = np.sqrt(np.diag(nCov))
			nCov = nCov / np.outer(std_, std_)	
			return nCov, sCov.shape[0]
		else:
			print("Cannot find %s" % null_model)
			return None
		
		
	def _MG2(self, sCov, sampleNum):
	
		w, v =  np.linalg.eig(sCov)
	
		rand_eig_max = np.power( (1 + sCov[0] / sampleNum), 2)
		null_comp = np.array(w<=rand_eig_max).ravel()
		v = v[:, null_comp]
		w = w[null_comp]
		R =  v @ np.diag(w) @ v.T
		return R, v.shape[0] * v.shape[1] + w.shape[0]

	def _MG3(self, sCov, sampleNum):
	
		w, v =  np.linalg.eig(sCov)
	
		rand_eig_max = np.power( (1 + sCov[0] / sampleNum), 2)
		null_comp = np.array((w<=rand_eig_max) | (w==np.max(w))).ravel()
		v = v[:, null_comp]
		w = w[null_comp]
		R =  v @ np.diag(w) @ v.T

		return R, v.shape[0] * v.shape[1] + w.shape[0]
