# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg
from scipy import sparse
from scipy import stats
import os
import tqdm
import sys
from functools import partial
from ._common import _fast_mat_inv_lapack
from ._common import _comp_EBIC
from ._common import _comp_loglikelihood


class iScola:

    def __init__(self):
        self.approximation = True

    input_matrix_type = "pres"

    def detect(self, C_samp, iC_null, lam, Winit = None):
        """
        Scola algorithm for precision matrices. 
            
        Parameters
        ----------
        C_samp : 2D numpy.ndarray, shape (N, N)
            Sample correlation matrix. 
        iC_null : 2D numpy.ndarray, shape (N, N)
            Null precision matrix.
        lam : float
            Lasso penalty.
    
        Returns
        -------
        W : 2D numpy.ndarray, shape (N, N)
            Weighted adjacency matrix of the generated network.
        """
	
        iC_samp = self._ridge(C_samp, 0.0001)
        N = C_samp.shape[0]
        mt = np.zeros((N, N))
        vt = np.zeros((N, N))
        t = 0
        eps = 1e-8
        b1 = 0.9
        b2 = 0.999
        maxscore = -1e300
        t_best = 0
        eta = 0.001
        maxIteration = 1e7
        maxLocalSearch = 300
        Lambda = 1 / (np.power(np.abs(iC_samp - iC_null), 2) + 1e-20)
        np.fill_diagonal(Lambda, 0)
        
        if self.approximation:
            W = self._prox(iC_samp - iC_null, lam * Lambda)
            return W

        if Winit is not None:
            W = Winit 
        else:
            W = self._prox(iC_samp - iC_null, lam * Lambda)

        score0 = self._comp_penalized_loglikelihood(W, C_samp, iC_null, Lambda)
        _diff_min = 1e300
        while (
            (t < maxIteration) & ((t - t_best) <= maxLocalSearch) & (_diff_min > 5e-5)
        ):
            t = t + 1
            gt = self._calc_gradient(C_samp, iC_null, W)

            mt = b1 * mt + (1.0 - b1) * gt
            vt = b2 * vt + (1.0 - b2) * np.power(gt, 2)
            mthat = mt / (1.0 - np.power(b1, t))
            vthat = vt / (1.0 - np.power(b2, t))
            dtheta = np.divide(mthat, (np.sqrt(vthat) + eps))

            W_prev = W
            W = self._prox(W - eta * dtheta, eta * lam * Lambda)
            _diff = np.max(np.abs(W - W_prev))
            if _diff < _diff_min:
                _diff_min = _diff
                t_best = t

            if _diff < 5e-5:
                break

            # If the score isn't improved in the first 50 iterations, then break
            if t % 10 == 0:
                score = self._comp_penalized_loglikelihood(W, C_samp, iC_null, Lambda)
                #score = self._comp_penalized_loglikelihood(W, C_samp, C_null, Lambda)
                if (prev_score > score):
                    break
                prev_score = score
            if t % 50 == 0:
                score = self._comp_penalized_loglikelihood(W, C_samp, iC_null, Lambda)
                if (score0 > score):
                    break
        return W

    def _ridge(self, C_samp, rho):
        """
        Compute the precision matrix from covariance matrix with a ridge regularization.
        
        Parameters
        ----------
        C_samp : 2D numpy.ndarray, shape (N, N)
            Sample covariance matrix
        rho : float
            Regularization parameter

        Returns
        -------
        iC : 2D numpy.ndarray, shape (N, N)
            Precision matrix
        """
        w, v = np.linalg.eigh(C_samp)
        lambda_hat = 2 / (np.sqrt(w ** 2) + np.sqrt(w ** 2 + 8 * rho))
        iC = np.matmul(np.matmul(v, np.diag(lambda_hat)), v.T)
        return iC 

    def comp_upper_lam(self, C_samp, iC_null):
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
        iC_samp = self._ridge(C_samp, 0.0001)
        absCov = np.abs(iC_samp - iC_null)
        D = iC_null - iC_samp
        v = np.triu(np.multiply(np.abs(D), np.power(absCov, 2)), 1)
        nnz = np.nonzero(v)
        K = len(nnz[0])
        if len(nnz) == 0:
            return 1
        else:
            lam_upper = np.sort(v[nnz])[np.floor(K * 0.99).astype(int)]
        return lam_upper

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

    def _calc_gradient(self, sCov, inv_nCov, dCov):
        Cov = _fast_mat_inv_lapack(inv_nCov + dCov)
        g = sCov - Cov
        g = (g + g.T) / 2
        g = np.nan_to_num(g)
        return g

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
            _comp_loglikelihood(W, C_samp, C_null, "pres")
            - np.sum(np.multiply(Lambda, np.abs(W))) / 4
        )
