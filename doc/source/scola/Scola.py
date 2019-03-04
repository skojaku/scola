import numpy as np
from scipy import linalg
from scipy import sparse
from scipy import stats
import pickle
import os
import tqdm
import sys
import configcorr as cg
from functools import partial

class Scola():
	"""Scola algorithm
	
	An algorithm to construct networks from correlaton matrices

	.. role:: python(code)
    		:language: python

	"""

	
	def __init__(self, path_to_working_directory):
		"""
	
		Parameters
		----------
		path_to_working_directory: string
			Path to the working directory. 
			All results will be saved in this directory.
			If the directory does not exist, the program will create the directory.  
		"""

		self.path_to_working_directory = path_to_working_directory

	# Main routine	
	def construct_network(self, C_samp, L, null_model = "config", lambda_list = np.logspace(-10, 0, 30)[::-1], beta = 0.5, disp=True):	
		""" Construct a network from a correlation matrix
		
		Parameters	
		----------
		C_samp: NxN numpy.matrix 
			Pearson correlation matrix for the input data. 
			N is the number of nodes.  
		
		L: int
			Number of samples.
		
		null_model: string; Optional; Default = "config"
			Name of null model. See the Readme for the available null models.

		lambda_list: float numpy.array (Optinal; Default = numpy.logspace(-10, 0, 30)[::-1])
			The lasso penalties.
			The algorithm will generate networks with the given lasso penalties. 
			Then, it outputs the network that minimises the extended BIC.  
			
		beta: float (Optional; Default = 0.5; float; 0 <= beta <=1)
			Parameter of the extended BIC.
			Larger beta would yield a sparser network. 
	
		disp: boolean (Optional; Default=True)
			Set to True to print the progress.
	
	
		Return	
		------
		W: NxN numpy.matrix
			Weighted adjacency matrix of the constructed network.
			
		EBIC: float
			The value of the extended Bayesian Information Criterion. 
		
		lam: float
			The lasso penalty that yields the minimum EBIC value among those in lambda_list.
		"""

		self.disp = disp
		self.null_model = null_model
	
		# Create a directory to save results if it does not exist.	
		if not os.path.exists(self.path_to_working_directory):
			os.mkdir(self.path_to_working_directory)
	
		# The tentative results are saved in <working directory>/networks-<name of the null model>.pickle
		file_tentative_results = '%s/networks-%s.pickle' % (self.path_to_working_directory, null_model)
		networks, Lams = self._load_results(file_tentative_results) 
	
		# Compute the null correlation matrices
		C_null, num_parameters_null = self._compute_null_model(C_samp, null_model, self.disp)

		# Generate a network for each lambda value in lambda_list 
		N = C_samp.shape[0] 
		I = float(len(lambda_list))
		if self.disp:
			print("")
			print('Constructing networks...')
		for lam in tqdm.tqdm(lambda_list, disable=(self.disp==False)):
	
			if lam not in Lams: # if lam has not been scanned yet

				# Construct a network from the given correlation matrix	
				W = self._construct_network(C_samp, C_null, lam)
		
				# Compute the extended BIC for the generated network	
				EBIC = self._calc_EBIC(W, C_samp, C_null, L, beta, num_parameters_null)
	
				# Save the generated network
				networks+=[{'W':W, 'EBIC':EBIC, 'lam':lam}]
				self._save_results(file_tentative_results, networks)
			
			else:
				net = networks[np.where(Lams == lam)[0][0]]
				W = net["W"]
				EBIC = net["EBIC"]
		
		if self.disp:
			print("")
			print('Finished')
	
		return self._find_best_model(networks)

	def get_network(self):
		""" Get the generated network
			
		Return	
		------
		W: NxN numpy.matrix
			Weighted adjacency matrix of the constructed network.
		"""

		file_tentative_results = '%s/networks-%s.pickle' % (self.path_to_working_directory, self.null_model)
		networks, Lams = self._load_results(file_tentative_results) 
		EBICs = [net["EBIC"] for net in networks]	
		idx = np.argmin(np.array(EBICs))
		return networks[idx]["W"]

	def get_null_corr_matrix(self):
		""" Get the null correlation matrix 
			
		Return	
		------
		Cnull: NxN numpy.matrix
			Null correlation matrix.	
		"""

		file_null_model = '%s/null-model-%s.pickle' % (self.path_to_working_directory, self.null_model)
		with open(file_null_model, 'rb') as f:
			res = pickle.load(f)
		C_null = res["C_null"]
		num_parameters_null = res["num_parameters_null"]
		return C_null, num_parameters_null	

	# Private function
	def _find_best_model(self, networks):
		EBICs = [net["EBIC"] for net in networks]	
		idx = np.argmin(np.array(EBICs))
		return networks[idx]["W"], networks[idx]["EBIC"], networks[idx]["lam"]

	# Private function
	def _compute_null_model(self, C_samp, null_model, disp):

		file_null_model = '%s/null-model-%s.pickle' % (self.path_to_working_directory, null_model)
		C_null = []
		num_parameters_null = -1
		
		if disp:
			print('Computing null correlation matrix.')

		if os.path.exists(file_null_model):
			with open(file_null_model, 'rb') as f:
				res = pickle.load(f)
				C_null = res["C_null"]
				num_parameters_null = res["num_parameters_null"]
		else:
			if null_model == "white-noise":	
				C_null = np.matrix(np.eye(C_samp.shape[0]))
				num_parameters_null = 0
			elif null_model == "hqs":	
				C_null = np.mean(np.triu(C_samp,1)) * np.ones(C_samp.shape)
				np.fill_diagonal(C_null, 1) 
				num_parameters_null = 1
			elif null_model == "config":
				C_null = cg.max_ent_config_dmcc(C_samp, 1e-4, disp)
				std_ = np.sqrt(np.diag(C_null))
				C_null = C_null / np.outer(std_, std_)
				num_parameters_null = 	C_samp.shape[0]
			else:
				raise ValueError('Unknown null model. See the README for the available null models.')
	
			with open(file_null_model, 'wb') as f:
				pickle.dump({'C_null':C_null, 'num_parameters_null':num_parameters_null},f)
		
		return C_null, num_parameters_null
		
	# Private function
	def _construct_network(self, C_samp, C_null, lam):
		N = C_samp.shape[0]
		W = self._prox(C_samp- C_null, lam)
		score_prev = -1.7976931348623157e+308
		while True:
			_W = self._maximisation_step(C_samp, C_null, W, lam)
			score  = self._penalized_likelihood(_W, C_samp, C_null, lam)
			if score <= score_prev:
				break
			W = _W
			score_prev = score
		
		return W
		
	# Private function
	def _maximisation_step(self, C_samp, C_null, W_base, lam):
	 
		# Initialise the variables for the ADAM algorithm
		N = C_samp.shape[0]
		mt = np.matrix(np.zeros((N, N)))
		vt = np.matrix(np.zeros((N, N)))
		t = 0
		eps = 1e-8
		b1 = 0.9
		b2 = 0.99
		maxscore = -1.7976931348623157e+308
		t_best = 0
		eta = 0.001
		maxIteration = 1e+7
		maxLocalSearch = 300
		quality_assessment_interval = 100
		W_best = W_base
		W = W_base
	
		# Compute the penalty strength using the adaptive lasso
		Lambda = 1/(np.power(np.abs(C_samp-C_null),2)+1e-20)

		inv_C_base = self._fast_inv_mat_lapack(C_null + W_base)
		while (t < maxIteration) & (t <= (t_best + maxLocalSearch+1)):
			t = t + 1
			# Calculate the gradient
			inv_C = self._fast_inv_mat_lapack(C_null+W) # Call LAPAC for fast matrix inversions
			gt = inv_C_base - inv_C @ C_samp@ inv_C
			gt = (gt + gt.T)/2
			gt = np.nan_to_num(gt)
			np.fill_diagonal(gt, 0)
			mt = b1 * mt + (1-b1)*gt
			vt = b2 * vt + (1-b2)*np.power(gt, 2)
			mthat = mt / (1-np.power(b1,t))
			vthat = vt / (1-np.power(b2,t))
			dtheta = np.divide(mthat, (np.sqrt(vthat) + eps))
		
			# Update the network
			W = self._prox(W - eta * dtheta, eta * lam * Lambda)
			np.fill_diagonal(W, 0)

			# Compute the likelihood 
			if (t % quality_assessment_interval) == 0:
				s  = self._penalized_likelihood(W, C_samp, C_null, lam)
				if s > maxscore:
					W_best = W
					maxscore = s
					t_best = t

		return W_best

	# Private function
	def _fast_inv_mat_lapack(self, M):
		zz , _ = linalg.lapack.dpotrf(M, False, False)
		inv_M , info = linalg.lapack.dpotri(zz)
		inv_M = np.triu(inv_M) + np.triu(inv_M, k=1).T
		return inv_M
		
	# Private function
	def _loglikelihood(self, W, C_samp, C_null):
		Cov = W + C_null
		w, v = np.linalg.eig(Cov)
		if np.min(w) < 0:
			v = v[:,w>0]
			w = w[w>0]
			iCov = np.real(v @ np.diag(1/w) @ v.T)
			return -0.5 *  np.prod(w) - 0.5*np.trace(C_samp@ iCov) - 0.5 * Cov.shape[0] * np.log(2 * np.pi)
		s, v = np.linalg.slogdet(Cov)
		return -0.5 *  s * v  - 0.5*np.trace(C_samp@ linalg.inv(Cov)) - 0.5 * Cov.shape[0] * np.log(2 * np.pi)
	
	# Private function
	def _calc_EBIC(self, W, C_samp, C_null, L, beta, null_model_parameters):
		k= null_model_parameters + np.sum(np.abs(W)>1e-30)/2
		EBIC = np.log(L) * k - 2 * L * self._loglikelihood(W, C_samp, C_null)
		EBIC+= 4 * beta * k * np.log(W.shape[0])
		return EBIC 

	# Private function
	def _prox(self, y, lam):
		return np.multiply(np.sign(y),  np.maximum(np.abs(y) - lam, np.matrix(np.zeros(y.shape))))

	# Private function
	def _penalized_likelihood(self, W, C_samp, C_null, lam):
		return self._loglikelihood(W, C_samp, C_null)  - lam * np.sum(np.abs(W)) 

	# Private function
	def _save_results(self, filename, networks):
		with open(filename, 'wb') as f:	
			pickle.dump(networks, f)
		
	# Private function
	def _load_results(self, filename):
		if os.path.exists(filename): # if tentative-results.pickle exists
			with open(filename, 'rb') as f:	
				networks = pickle.load(f)
			Lams = [network["lam"] for network in networks]	
		else:
			networks = []
			Lams = []
		return networks, Lams 
