import numpy as np
from scipy import linalg
from scipy import stats
from functools import partial

def optimize_threshold(cvFilter, sCov, nCov, sampleNum, alphalist = None, disp=True, null_model= "config", gamma = 1):	

	# Setting candidate threshold values	
	if alphalist is None:
		alphalist = cvFilter.candidateThreshold(sCov, nCov, num=30)
	else:
		alphalist = alphalist

	# Complexity of null model
	num_null_model_param = 0 
	if null_model == "config":
		num_null_model_param = 2 * nCov.shape[0] 
	elif null_model == "hqs":
		num_null_model_param = 2 
	elif null_model == "eye":
		num_null_model_param = nCov.shape[0]
	
	# Scan each candidate threshold 	
	N = sCov.shape[0] 
	dCov = np.matrix(np.zeros((N, N)))
	dCovList_ = []
	dCov_ = None
	bic_ = 1e+30 
	if disp:
		print('  \talpha  \tDensity\tp-value\tBIC')
	for aid, alpha in enumerate(alphalist):

		dCov = cvFilter.fit(sCov, nCov, dCov, alpha)

		density = np.sum(np.abs(dCov)>1e-30)/ (N * (N - 1))
		pvalue = _likelihood_ratio_test(dCov, sCov, nCov, sampleNum)
		bic = _calc_bic(dCov, sCov, nCov, sampleNum, gamma, num_null_model_param)
	
		mark = ""	
		if bic_ > bic:
			dCov_ = dCov
			alpha_ = alpha
			pvalue_ = pvalue 
			bic_ = bic
			mark = " (*)"
		
		if disp:
			density = np.sum(np.abs(dCov)>0) / (N * (N-1))
			print("%2d\t%f\t%f\t%f\t%f%s" % (aid, alpha, density, pvalue, bic, mark))	

		dCovList_ += [{'dCov':dCov, 'alpha':alpha}]
	
	return {'dCov':dCov_, 'alpha':alpha_, 'pvalue':pvalue_, 'bic':bic_, 'gamma':gamma, 'dCovList':dCovList_, 'alphaList':alphalist, 'null_model':null_model}



# Log likelihood
def _loglikelihood(dCov, sCov, nCov):
	Cov = dCov + nCov
	if np.linalg.cond(Cov) < 1e-30:
		return -1e+30
	s, v = np.linalg.slogdet(Cov)
	return -0.5 *  s * v  - 0.5*np.trace(sCov @ linalg.inv(Cov)) - 0.5 * Cov.shape[0] * np.log(2 * np.pi)

# Compute (extended) Baysian Information Criterion
def _calc_bic(dCov, sCov, nCov, sampleNum, gamma, null_model_complexity = 0):
	k = null_model_complexity
	k+= np.sum(np.abs(dCov)>1e-30)/2
	bic = np.log(sampleNum) * k - 2 * sampleNum * _loglikelihood(dCov, sCov, nCov)
	bic+= 4 * gamma * k * np.log(dCov.shape[0])
	return bic 
	
# Likelihood ratio test	
def _likelihood_ratio_test(dCov, sCov, nCov, sampleNum):
	c = -2 * sampleNum * (_loglikelihood(dCov * 0, sCov, nCov) - _loglikelihood(dCov, sCov, nCov))
	k= np.sum(np.abs(dCov)>1e-30) / 2
	if k < 1:
		return 1
	pval = 1 - np.exp(stats.chi2.logcdf(c, k))
	return pval
