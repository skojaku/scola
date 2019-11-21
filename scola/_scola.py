# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg
from scipy import sparse
from scipy import stats
import os
import tqdm
import sys
from functools import partial
from . import gradient_descent as gd
from ._common import _fast_mat_inv_lapack
from ._common import _comp_EBIC
from ._common import _comp_loglikelihood


class Scola:

    def __init__(self):
        self.approximation = False
        self.input_matrix_type = "cov"

    def detect(self, C_samp, C_null, lam, Winit = None):
        """
	    Minorisation-maximisation algorithm. 
	        
	    Parameters
	    ----------
	    C_samp : 2D numpy.ndarray, shape (N, N)
	        Sample correlation matrix. 
	    C_null : 2D numpy.ndarray, shape (N, N)
	        Null correlation matrix.
	    lam : float
	        Lasso penalty.
	
	    Returns
	    -------
	    W : 2D numpy.ndarray, shape (N, N)
	        Weighted adjacency matrix of the generated network.
	    """

        N = C_samp.shape[0]
        Lambda = 1.0 / (np.power(np.abs(C_samp - C_null), 2) + 1e-20)

        if self.approximation:
            W = self._prox(C_samp - C_null, lam * Lambda)
            return W

        if Winit is not None:
            W = Winit 
        else:
            W = self._prox(C_samp - C_null, lam * Lambda)

        score_prev = -1e300
        count_small_itnum = 0 
        score = self._comp_penalized_loglikelihood(W, C_samp, C_null, lam * Lambda)
        while True:
            _W, itnum = self._maximisation_step(C_samp, C_null, W, lam)
            score = self._comp_penalized_loglikelihood(_W, C_samp, C_null, lam * Lambda)
           
            #print(score, itnum)
            if score <= score_prev:
                break
            W = _W

            # The changes in W are gentle, stop iteration.
            if itnum <=2:
               count_small_itnum+=1 
               if count_small_itnum>5:
                    break
            else:
               count_small_itnum=0 

            score_prev = score

        return W

    def comp_upper_lam(self, C_samp, C_null):
        """
        Compute the upper bound of the Lasso penalty.
    
        Parameters
        ----------
        C_samp : 2D numpy.ndarray, shape (N, N)
            Sample correlation matrix. 
        C_null : 2D numpy.ndarray, shape (N, N)
            Null correlation matrix used for constructing the network.
    
        Returns
        -------
        lam_upper : float
            Upper bound of the Lasso penalty. 
        """

        abC_samp = np.abs(C_samp - C_null)
        iCov = _fast_mat_inv_lapack(C_null)
        D = iCov - np.matmul(np.matmul(iCov, C_samp), iCov)
        lam_upper = np.max(np.multiply(np.abs(D), np.power(abC_samp, 2)))
        return lam_upper

    def _comp_penalized_loglikelihood(self, W, C_samp, C_null, Lambda):
        """
	    Compute the penalized log likelihood for a network. 
	    
	    Parameters
	    ----------
	    W : 2D numpy.ndarray, shape (N, N)
	        Weighted adjacency matrix of a network.
	    C_samp : 2D numpy.ndarray, shape (N, N)
	        Sample correlation matrix. 
	    C_null : 2D numpy.ndarray, shape (N, N)
	        Null correlation matrix used for constructing the network.
	    Lambda : 2D numpy.ndarray, shape (N, N)
	        Lambda[i,j] is the Lasso penalty for W[i,j]. 
	
	    Returns
	    -------
	    l : float
	        Penalized log likelihood for the generated network. 
	    """
        return (
            _comp_loglikelihood(W, C_samp, C_null, "cov")
            - np.sum(np.multiply(Lambda, np.abs(W))) / 4
        )

    def _prox(self, x, lam):
        """
	    Soft thresholding operator.
	    
	    Parameters
	    ----------
	    x : float
	        Variable.
	    lam : float
	        Lasso penalty.
	
	    Returns
	    -------
	    y : float
	        Thresholded value of x. 
	    """

        return np.multiply(np.sign(x), np.maximum(np.abs(x) - lam, np.zeros(x.shape)))

    def _maximisation_step(self, C_samp, C_null, W_base, lam):
        """
	    Maximisation step of the MM algorithm. 
	    (A subroutine for detect) 
	    ----------
	    C_samp : 2D numpy.ndarray, shape (N, N)
	        Sample correlation matrix. 
	    C_null : 2D numpy.ndarray, shape (N, N)
	        Null correlation matrix.
	    W_base : 2D numpy.ndarray, shape (N, N)
	        W at which the minorisation is performed.  
	    lam : float
	        Lasso penalty.
	
	    Returns
	    -------
	    W : 2D numpy.ndarray, shape (N, N)
	        Weighted adjacency matrix of the generated network.
	"""
        N = C_samp.shape[0]
        t = 0
        #eps = 1e-8
        #b1 = 0.9
        #b2 = 0.999
        maxscore = -1e300
        t_best = 0
        #eta = 0.001
        maxLocalSearch = 30
        maxIteration = 300 

        #maxIteration = 1e7
        #maxLocalSearch = 300
        W = W_base
        Lambda = lam / (np.power(np.abs(C_samp - C_null), 2) + 1e-20)
        inv_C_base = _fast_mat_inv_lapack(C_null + W_base)
        _diff_min = 1e300
        score0 = self._comp_penalized_loglikelihood(W, C_samp, C_null, Lambda)

        #adam = gd.ADABound()
        adam = gd.ADAM()
        adam.eta = 0.01
        adam.decreasing_learning_rate = True
        prev_score = score0
        while (
            (t < maxIteration) & ((t - t_best) <= maxLocalSearch) & (_diff_min > 1e-5)
        ):
            t = t + 1
            inv_C = _fast_mat_inv_lapack(C_null + W)
            gt = inv_C_base - np.matmul(np.matmul(inv_C, C_samp), inv_C)

            W_prev = W
            np.fill_diagonal(gt, 0.0)
            W = adam.update( W, -gt, Lambda)    
            np.fill_diagonal(W, 0.0)

            _diff = np.sqrt( np.mean(np.power(W - W_prev, 2)) )
       

            if _diff < (_diff_min * 0.95):
                _diff_min = _diff
                t_best = t

            if _diff < 5e-5:
                break

            # If the score isn't improved in the first 50 iterations, then break
            if t % 10 == 0:
                score = self._comp_penalized_loglikelihood(W, C_samp, C_null, Lambda)
                if (prev_score > score):
                    break
                prev_score = score
            if t % 50 == 0:
                score = self._comp_penalized_loglikelihood(W, C_samp, C_null, Lambda)
                if (score0 > score):
                    break

        return W, t
