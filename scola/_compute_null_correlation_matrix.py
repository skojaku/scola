# -*- coding: utf-8 -*-
import numpy as np
from ._common import _comp_EBIC 
from ._common import _comp_loglikelihood 
from ._common import _fast_mat_inv_lapack

def _compute_null_correlation_matrix(C_samp, null_model):
    """
        Compute a null correlation matrix for a sample correlation matrix.
            
        Parameters
        ----------
        C_samp : 2D numpy.ndarray, shape (N, N)
            Sample correlation matrix.
        null_model : str
            Name of the null model.
    
        Returns
        -------
        C_null : 2D numpy.ndarray, shape (N, N)
            Estimated null correlation matrix.
        K_null : int
            Number of parameters of the null model. 
        """

    C_null = []
    K_null = -1
    if null_model == "white-noise":
        C_null = np.eye(C_samp.shape[0])
        K_null = 0
    elif null_model == "hqs":
        C_null = np.mean(np.triu(C_samp, 1)) * np.ones(C_samp.shape)
        np.fill_diagonal(C_null, 1)
        K_null = 1
    elif null_model == "config":
        C_null = _estimate_configuration_model(np.array(C_samp), 1e-4)
        std_ = np.sqrt(np.diag(C_null))
        C_null = C_null / np.outer(std_, std_)
        K_null = C_samp.shape[0]
    else:
        raise ValueError(
            "Null model %s is unknown. See the Readme for the available null models."
            % null_model
        )
    return C_null, K_null

def _estimate_configuration_model(C_samp, tolerance=1e-5):
    """
        Compute the configuration model for correlation matrices
        using the gradient descent algorithm.
        
        Parameters
        ----------
        C_samp : 2D numpy.ndarray, shape (N, N)
            Sample correlation matrix.
        tolerance: float
            Tolerance in relative error.
    
        Returns
        -------
        C_con : 2D numpy.ndarray, shape (N, N)
            The correlation matrix under the configuration model that
            preserves the row sum (and column sum) of C_samp as expectation.
        """

    cov = np.asanyarray(C_samp)
    std_ = np.sqrt(np.diag(cov))
    _C_samp = cov / np.outer(std_, std_)

    N = _C_samp.shape[0]
    s = np.sum(_C_samp, axis=1)
    K = np.linalg.pinv(_C_samp)

    theta = np.concatenate([np.diag(K), np.zeros(N)])
    mt = np.zeros(2 * N)
    vt = np.zeros(2 * N)
    t = 0
    eps = 1e-8
    b1 = 0.9
    b2 = 0.999
    t_best = 0
    eta = 0.001
    maxIteration = 1e7
    while t < maxIteration:
        t = t + 1

        K_est = np.add.outer(theta[N : 2 * N], theta[N : 2 * N]) + np.diag(
            theta[0:N]
        )
        C_con = _fast_mat_inv_lapack(K_est)

        error = np.max(np.abs(np.sum(C_con, 1) - s) / s)
        if error < tolerance:
            break

        dalpha = np.diag(C_con) - np.diag(_C_samp)
        dbeta = (2.0 / N) * (np.sum(C_con, axis=1) - s)

        gt = np.concatenate([dalpha, dbeta])
        mt = b1 * mt + (1.0 - b1) * gt
        vt = b2 * vt + (1.0 - b2) * np.power(gt, 2.0)
        mthat = mt / (1.0 - np.power(b1, t))
        vthat = vt / (1.0 - np.power(b2, t))
        dtheta = np.divide(mthat, (np.sqrt(vthat) + eps))

        theta = theta + eta * dtheta

    return C_con
