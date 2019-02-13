import numpy as np
from scipy import linalg
from scipy import stats
from functools import partial
import time
from scipy import sparse

class LassoFilter():
	
	def __init__(self, tol = 1e-5, eta = 0.001, maxIteration = 1e+7, maxLocalSearch = 100, disp=True):
		self.tol = tol

		self.eta = eta
		self.maxIteration = maxIteration
		self.maxLocalSearch = maxLocalSearch
		self.disp = disp

	def _maximize(self, sCov, nCov, dCov_prev, alpha, disp=False):
	#def _maximize(self, sCov, nCov, dCov_prev, alpha, disp=True):
		N = sCov.shape[0]	
		dCov = dCov_prev 
		dCov_old = dCov
		
		fill_diag_zeros = True
		if np.max(np.abs(np.diag(sCov - nCov))) >1e-7:
			fill_diag_zeros = False
		#
		# ADAM gradient descent algorithm
		#
		mt = np.matrix(np.zeros((N, N)))
		vt = np.matrix(np.zeros((N, N)))
		t = 0
		eps = 1e-8
		b1 = 0.9
		b2 = 0.99
		dif = 1
		maxscore = -1.7976931348623157e+308
		t_best = 0
		dCov_best = dCov_prev
		_diff = np.matrix(np.zeros((N, N)))
		W = 1/(np.power(np.abs(sCov-nCov),2)+1e-20)

		if disp:
			print('Iteration\tLikelihood\tDiff')

		ts = time.time()
		iCov_prev = self._fast_inv_mat_lapack(nCov + dCov_prev)
		iCov_guess = iCov_prev 
		while (t < self.maxIteration) & (t <= (t_best + self.maxLocalSearch+1)):

			t = t + 1

			gt, iCov_guess = self._calc_gradient(dCov, dCov_prev, sCov, nCov, iCov_prev, iCov_guess)
			if fill_diag_zeros:
				np.fill_diagonal(gt, 0)
			#gt = gt + alpha * np.linalg.multi_dot([W , np.sign(dCov)]) 
	
			mt = b1 * mt + (1-b1)*gt
			vt = b2 * vt + (1-b2)*np.power(gt, 2)
			mthat = mt / (1-np.power(b1,t))
			vthat = vt / (1-np.power(b2,t))
			dtheta = np.divide(mthat, (np.sqrt(vthat) + eps))

			dCov = self._prox(dCov - self.eta * dtheta, self.eta * alpha * W)
			if fill_diag_zeros:
				np.fill_diagonal(dCov, 0)

			if (t % 100) == 0:
				s  = self._score(dCov, sCov, nCov, alpha)
				tmp = ""	
				if s > maxscore:
					dCov_best = dCov
					maxscore = s
					t_best = t
					tmp = " *"
				if disp:
					tf = time.time()
					print('%9d\t%f%s\t%f' %(t, s, tmp, tf-ts))
					ts = tf

		return dCov_best
		
		
	def fit(self, sCov, nCov, dCov_init, sampleNum, alpha, disp=False):
	#def fit(self, sCov, nCov, dCov_init, alpha, disp=False):
	
		N = sCov.shape[0]
		
		dCov_init = self._prox(sCov - nCov, alpha)
		dCov = dCov_init 
		dCov_prev = dCov_init
		dCov_old = dCov
		dCov_best = []
		t = 0	
		t_best = 0
		ts = time.time()
		maxscore = -1.7976931348623157e+308
		while True:

			t = t + 1
			
			# Maximize
			dCov = self._maximize(sCov, nCov, dCov, alpha)

			s  = self._score(dCov, sCov, nCov, alpha)
			if s > maxscore:
				dCov_best = dCov
				maxscore = s
				t_best = t
				tmp = " *"
			else:
				break
			if disp:
				tf = time.time()
				print('%9d\t%f%s\t%f' %(t, s, tmp, tf-ts))
				ts = tf
		return dCov_best

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
	
	def candidateThreshold(self, sCov, nCov, num=30):
		return np.logspace(-10, 0, num)[::-1]	
	
	def _select_penaltyFunction(self, sCov, nCov, param=None):
		
		if self.penalty == "lasso":
			return lambda alpha, X : alpha * np.ones(X.shape)

		elif self.penalty == "adaptive_lasso":
			if param is None:
				param = 2

			def adaptive_lasso_penalty(alpha, X, param):
				g = alpha * 1/np.power(np.abs(sCov-nCov)+ 1e-20, param)
				return g	

			return partial(adaptive_lasso_penalty, param = param) 

		elif self.penalty == "scad":
			def scad_penalty(alpha, X, param = 3.7):
				I = (np.abs(X)<=alpha).astype(int)
				g = alpha * ( I + np.multiply( np.maximum(param * alpha - np.abs(X), np.zeros(X.shape)) / ((param - 1) * alpha), 1-I ) )
				np.fill_diagonal(g, 0) 
				return g	
			return scad_penalty

	def _fast_inv_mat_lapack(self, M):
		zz , _ = linalg.lapack.dpotrf(M, False, False)
		inv_M , info = linalg.lapack.dpotri(zz)
		# lapack only returns the upper or lower triangular part 
		inv_M = np.triu(inv_M) + np.triu(inv_M, k=1).T
		return inv_M

	def _calc_gradient(self, dCov, dCov_prev, sCov, nCov, iCov_prev, iCov_guess):
		#iCov = self._fast_inv_mat_lapack(nCov + dCov)
		if iCov_prev is None:
			iCov_prev = self._fast_inv_mat_lapack(nCov + dCov_prev)
		#iCov = linalg.inv(nCov + dCov) 
		#iCov_prev = linalg.inv(nCov + dCov_prev) 
		#iCov = self._fast_inv_mat_lapack(nCov + dCov)
#		if dCov.shape[0] < 1000:
#			iCov, solved = self._inv_approx(nCov + dCov, iCov_guess)
#			if solved == False:
#				iCov = self._fast_inv_mat_lapack(nCov + dCov)
#		else:
#			iCov, solved = self._inv_approx(nCov + dCov, iCov_guess)
		iCov = self._fast_inv_mat_lapack(nCov + dCov)
		#iCov = self._inv_approx(nCov + dCov, iCov_guess)
			
		g = iCov_prev - iCov @ sCov @ iCov
		
		# for numerical stability
		g = (g + g.T)/2
		g = np.nan_to_num(g)
		return g, iCov

	def _prox(self, y, alpha):
		return np.multiply(np.sign(y),  np.maximum(np.abs(y) - alpha, np.matrix(np.zeros(y.shape))))


	def _score(self, dCov, sCov, nCov, alpha):
		return self.loglikelihood(dCov, sCov, nCov)  - alpha * np.sum(np.abs(dCov)) 

