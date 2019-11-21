# -*- coding: utf-8 -*-
from scipy import linalg
from scipy import sparse
from scipy import stats
import numpy as np

def _remedy_degeneracy(C_samp, rho = 1e-3):
    w, v = np.linalg.eigh(C_samp)
    if np.min(w) < rho:

        w[w<0] = rho

        # Compute the precision matrix from covariance matrix with a ridge regularization.
        lambda_hat = 2 / (np.sqrt(w ** 2) + np.sqrt(w ** 2 + 8 * rho))
        iC = np.matmul(np.matmul(v, np.diag(lambda_hat)), v.T)

        # Compute the correlation matrix from the precision matrix
        _C_samp = np.linalg.inv(iC)
        
        # Homogenize the variance 
        std_ = np.sqrt(np.diag(_C_samp))
        C_samp = _C_samp / np.outer(std_, std_)

    return C_samp

def _penalized_inverse(C_samp, rho = 1e-3):
    w, v = np.linalg.eigh(C_samp)
    lambda_hat = 2 / (np.sqrt(w ** 2) + np.sqrt(w ** 2 + 8 * rho))
    iC = np.matmul(np.matmul(v, np.diag(lambda_hat)), v.T)
    return iC 

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


def _comp_EBIC(W, C_samp, C_null, L, beta, Knull, input_matrix_type):
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
        input_matrix_type: string
	    Type of matrix to be given (covariance or precision)
    
        Returns
        -------
        EBIC : float
            The extended BIC value for the generated network.
        """
    k = Knull + np.count_nonzero(np.triu(W, 1)) / 2 + np.count_nonzero(np.diag(W))
    EBIC = (
        np.log(L) * k
        - 2 * L * _comp_loglikelihood(W, C_samp, C_null, input_matrix_type)
        + 4 * beta * k * np.log(W.shape[0])
    )
    return EBIC


def _comp_loglikelihood(W, C_samp, C_null, input_matrix_type):
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
        input_matrix_type: string
	    Type of matrix to be given (covariance or precision)
    
        Returns
        -------
        l : float
            Log likelihood for the generated network. 
        """
    if input_matrix_type == "cov":
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
    else:
        iCov = W + C_null
        w, v = np.linalg.eig(iCov)
        if np.min(w) < 0:
            v = v[:, w > 0]
            w = w[w > 0]
        l = (
            0.5 * np.sum(np.log(w))
            - 0.5 * np.trace(np.matmul(C_samp, iCov))
            - 0.5 * iCov.shape[0] * np.log(2 * np.pi)
        )

    return np.real(l)

