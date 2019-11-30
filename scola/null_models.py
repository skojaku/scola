# -*- coding: utf-8 -*-
import numpy as np
from ._common import _comp_EBIC
from ._common import _comp_loglikelihood
from ._common import _fast_mat_inv_lapack
from ._common import _penalized_inverse
from . import gradient_descent as gd
import warnings

def white_noise_model(C_samp):
    """
    Compute the white noise model for correlation matrices.
    
    Parameters
    ----------
    C_samp : 2D numpy.ndarray, shape (N, N)
        Sample correlation matrix.
    
    Returns
    -------
    C_null : 2D numpy.ndarray, shape (N, N)
        The correlation matrix under the white-noise model.
    K_null : int 
        Number of parameters to generate the null correlation matrix
    name : str
        Name of the null model ("white-noise")
    """
    C_null = np.eye(C_samp.shape[0])
    K_null = 0
    return C_null, K_null, "white-noise"

def hqs_model(C_samp):
    """
    Compute the HQS model for correlation matrices.
    
    Parameters
    ----------
    C_samp : 2D numpy.ndarray, shape (N, N)
        Sample correlation matrix.
    
    Returns
    -------
    C_null : 2D numpy.ndarray, shape (N, N)
        The correlation matrix under the HQS model.
    K_null : int 
        Number of parameters to generate the null correlation matrix
    name : str
        Name of the null model ("hqs")
    """
    C_null = np.mean(np.triu(C_samp, 1)) * np.ones(C_samp.shape)
    np.fill_diagonal(C_null, 1)
    K_null = 1
    return C_null, K_null, "hqs"

def configuration_model(C_samp, tolerance=5e-3):
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
    C_null : 2D numpy.ndarray, shape (N, N)
        The correlation matrix under the config model.
    K_null : int 
        Number of parameters to generate the null correlation matrix
    name : str
        Name of the null model ("config")
    """

    cov = np.asanyarray(C_samp)
    std_ = np.sqrt(np.diag(cov))
    _C_samp = cov / np.outer(std_, std_)

    #C_null = max_ent_config_dmcc(_C_samp, tolerance = 1e-5, transform_to_corr_mat = True)
    #s = np.array( np.sum(_C_samp, axis=1) ).reshape(-1)
    #_C_samp = _C_samp[s>0, :]
    #_C_samp = _C_samp[:, s>0]

    N = _C_samp.shape[0]
    s = np.sum(_C_samp, axis=1)

    # Approximated K
    Chqs, _, _ = hqs_model(_C_samp)
    K = np.linalg.inv(Chqs)
    offdiag_mean = np.mean( np.triu(K, k=1) ) * 2
    theta_0 = np.concatenate([np.diag(K) + offdiag_mean, np.ones(N) * offdiag_mean / 2])

    def fit(_C_samp, theta, eta):
        t = 0
        adam = gd.ADAM()
        #adam = gd.ADABound()
        adam.eta = eta 
    
        maxIteration = 1e7
        converged = False
        while t < maxIteration:
    
            t = t + 1
    
            K_est = np.add.outer(theta[N : 2 * N], theta[N : 2 * N]) + np.diag(theta[0:N])
            C_null = _fast_mat_inv_lapack(K_est)
    
            error = np.max(np.abs(np.sum(C_null, 1) - s) / s)
            if t < 10 or t % 100 == 0: 
                w, v = np.linalg.eigh(K_est)
                if np.min(w)<0:
                    break
            #print(error)
            if error < tolerance:
                converged = True
                break
    
            dalpha = np.diag(C_null) - np.diag(_C_samp)
            dbeta = (2.0 / N) * (np.sum(C_null, axis=1) - s)
    
            gt = np.concatenate([dalpha, dbeta])
            theta = adam.update( theta, gt, 0 )

        return C_null, converged

    eta = 0.01
    for trynum in range(40):
        C_null, converged = fit(_C_samp, theta_0, eta)
        if converged:
            break
        eta*=0.5

    
    if converged == False:
        warnings.warn("scola.nullmodels.configuration_model: Failed to converge. Try increase the tolerance value, e.g., the configuration_model(C_samp, tolerance = 1e-2).")


    std_ = np.sqrt(np.diag(C_null))
    C_null = C_null / np.outer(std_, std_)
    K_null = C_samp.shape[0]
    return C_null, K_null, "config"

# -*- coding: utf-8 -*-
#import cvxpy as cv
#import numpy as np
#
#def max_ent_config_dmcc(C, tolerance = 1e-5, transform_to_corr_mat = True):
#    """
#
#    DET-MAX algorithm for estimating the configuration model for correlation matrix data
#
#    Input
#      C: covariance matrix
#      tolerance: tolerance in relative error
#      transform_to_corr_mat = True if one transforms the input covariance matrix to the correlation matrix before running the gradient descent method. Otherwise False.
#      The default value is True.
#
#    Output
#      C_con: estimated covariance matrix
#
#    """
#
#    if transform_to_corr_mat == True: # Work on the correlation matrix
#        # Transform the original covariance matrix to the correlation matrix
#        cov = np.asanyarray(C)
#        std_ = np.sqrt(np.diag(cov))
#        _C = cov / np.outer(std_, std_)
#    else: #  work on the covariance matrix
#        _C = C
#
#    N = _C.shape[0] # Number of nodes
#
#    # Covariance matrix we will estimate
#    C_con = cv.Variable((N, N), PSD=True)
#
#    # Objective function to be maximized
#    objective = cv.Minimize(-cv.log_det(C_con))
#
#    # Constraints on C_con
#    constraints = [cv.sum(C_con, axis=0) == _C.sum(axis = 0), cv.diag(C_con) == np.diag(_C)]
#
#    # Optimization
#    prob = cv.Problem(objective, constraints)
#    prob.solve(verbose = True, eps = tolerance)
#
#    return C_con.value
