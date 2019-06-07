# -*- coding: utf-8 -*-
from scipy import linalg
from scipy import sparse
from scipy import stats
import numpy as np

def _fast_mat_inv_lapack(Mat):
    """
        Compute the inverse of a positive semidefinite matrix.
    
        This function exploits the positive semidefiniteness to speed up
        the matrix inversion.
            
        Parameters
        ----------
        Mat : 2D numpy.ndarray, shape (N, N)
            A positive semidefinite matrix.
    
        Returns
        -------
        inv_Mat : 2D numpy.ndarray, shape (N, N)
            Inverse of Mat.
        """

    zz, _ = linalg.lapack.dpotrf(Mat, False, False)
    inv_Mat, info = linalg.lapack.dpotri(zz)
    inv_Mat = np.triu(inv_Mat) + np.triu(inv_Mat, k=1).T
    return inv_Mat

def _comp_EBIC(W, C_samp, C_null, L, beta, Knull):
    """
        Compute the extended Bayesian Information Criterion (BIC) for a network. 
        
        Parameters
        ----------
        W : 2D numpy.ndarray, shape (N, N)
            Weighted adjacency matrix of a network.
        C_samp : 2D numpy.ndarray, shape (N, N)
            Sample correlation matrix.
        C_null : 2D numpy.ndarray, shape (N, N)
            Null correlation matrix used for constructing the network.
        L : int
            Number of samples.
        beta : float
            Parameter for the extended BIC. 
        K_null: int
            Number of parameters of the null correlation matrix.
    
        Returns
        -------
        EBIC : float
            The extended BIC value for the generated network.
        """

    k = Knull + np.count_nonzero(W) / 2
    EBIC = (
        np.log(L) * k
        - 2 * L * _comp_loglikelihood(W, C_samp, C_null)
        + 4 * beta * k * np.log(W.shape[0])
    )
    return EBIC

def _comp_loglikelihood(W, C_samp, C_null):
    """
        Compute the log likelihood for a network. 
        
        Parameters
        ----------
        W : 2D numpy.ndarray, shape (N, N)
            Weighted adjacency matrix of a network.
        C_samp : 2D numpy.ndarray, shape (N, N)
            Sample correlation matrix. 
        C_null : 2D numpy.ndarray, shape (N, N)
            Null correlation matrix used for constructing the network.
    
        Returns
        -------
        l : float
            Log likelihood for the generated network. 
        """

    Cov = W + C_null
    w, v = np.linalg.eig(Cov)
    if np.min(w) < 0:
        v = v[:, w > 0]
        w = w[w > 0]
    iCov = np.real(np.matmul(np.matmul(v, np.diag(1 / w)), v.T))
    l = (
        -0.5 * np.sum(np.log(w))
        - 0.5 * np.trace(np.matmul(C_samp, iCov))
        - 0.5 * Cov.shape[0] * np.log(2 * np.pi)
    )
    return np.real(l)
