# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg
from scipy import sparse
from scipy import stats
import os
import tqdm
import sys
from functools import partial


def generate_network(C_samp, L, null_model="all", disp=True):
    """
    Generate a network from a correlation matrix
    using the Scola algorithm.

    Parameters
    ----------
    C_samp : 2D numpy.ndarray, shape (N, N)
        Sample correlation matrix. N is the number of nodes.
    L : int
        Number of samples
    null_model : str, default 'all'
        Null model to be used for constructing the network.
        One can use the white noise model 
        (null_model='white-noise'), the Hirschberger-Qu-Steuer 
        model (null_model='hqs') or the configuration model 
        (null_model='config'). If null_model='all', then the 
        best one among the three null models in terms of the 
        extended Bayesian Information Criterion (BIC) is 
        selected.
    disp : bool, default True
        Set disp=True to display the progress of computation.
        Otherwise, set disp=False.

    Returns
    -------
    W : 2D numpy.ndarray, shape (N, N)
        Weighted adjacency matrix of the generated network.
    C_null : 2D numpy.ndarray, shape (N, N)
        Estimated null correlation matrix used for constructing the network.
    selected_null_model : str
        The null model selected by the Scola.
    EBIC : float
        The extended BIC value for the generated network.
    """

    if type(C_samp) is np.matrix:
        C_samp = np.array(C_samp)

    if type(C_samp) is not np.ndarray:
        raise TypeError("C_samp must be a numpy.ndarray")

    if (type(L) is not int) and (type(L) is not float):
        raise TypeError("L must be an integer")

    if type(disp) is not bool:
        raise TypeError("disp must be a bool")

    if type(null_model) is not str:
        raise TypeError("null_model must be a string")

    if null_model == "all":
        _null_models = ["white-noise", "hqs", "config"]
    else:
        _null_models = [null_model]

    # pbar is used for computing and displaying the progress of computation.
    pbar = tqdm.tqdm(disable=(disp is False), total=(13 * len(_null_models)))
    res = []
    for null_model in _null_models:
        W, C_null, EBIC_min = _golden_section_search(
            C_samp, L, null_model, beta=0.5, pbar, disp=disp
        )
        res += [[W, C_null, null_model, EBIC_min]]
    idx = np.argmin(np.array([r[3] for r in res]))
    pbar.close()
    return res[idx]


def _golden_section_search(C_samp, L, null_model, beta, pbar, disp):
    """
    Generate a network from a correlation matrix with 
    a golden-section search method

    Parameters
    ----------
    C_samp : 2D numpy.ndarray, shape (N, N)
        Sample correlation matrix.
    L : int
        Number of samples
    null_model : str
        Name of the null model.
    beta : float
        hyperparameter for the extended BIC.
    pbar : tqdm
        tqdm instance, which is used for computing and displaying 
        the progress of computation.
    disp : bool, default True
        Set disp=True to display the progress of computation.
        Otherwise, set disp=False.

    Returns
    -------
    W : 2D numpy.ndarray, shape (N, N)
        Weighted adjacency matrix of the generated network.
    C_null : 2D numpy.ndarray, shape (N, N)
        Estimated null correlation matrix used for constructing the 
        network.
    EBIC : float
        The extended BIC value for the generated network.
    """

    C_null, K_null = _compute_null_correlation_matrix(C_samp, null_model)

    lam_upper = _calc_upper_lam(C_samp, C_null)
    lam_lower = 0.0
    invphi = (np.sqrt(5) - 1.0) / 2.0
    invphi2 = (3.0 - np.sqrt(5.0)) / 2.0
    h = lam_upper - lam_lower
    lam_1 = lam_lower + invphi2 * h
    lam_2 = lam_lower + invphi * h
    n = int(np.ceil(np.log(0.01) / np.log(invphi)))
    N = C_samp.shape[0]
    W_best = None
    lam_best = 0
    EBIC_min = 0

    ns = pbar.n
    nf = ns + n

    for k in range(n):
        if k == 0:

            W_l = C_samp - C_null
            pbar.update()
            W_u = _MM_algorithm(C_samp, C_null, lam_upper)
            pbar.update()
            W_1 = _MM_algorithm(C_samp, C_null, lam_1)
            pbar.update()
            W_2 = _MM_algorithm(C_samp, C_null, lam_2)
            pbar.update()

            EBIC_l = _calc_EBIC(W_l, C_samp, C_null, L, beta, K_null)
            EBIC_u = _calc_EBIC(W_u, C_samp, C_null, L, beta, K_null)
            EBIC_1 = _calc_EBIC(W_1, C_samp, C_null, L, beta, K_null)
            EBIC_2 = _calc_EBIC(W_2, C_samp, C_null, L, beta, K_null)

            mid = np.argmin([EBIC_l, EBIC_u, EBIC_1, EBIC_2])
            W_best = [W_l, W_u, W_1, W_2][mid]
            lam_best = [lam_lower, lam_upper, lam_1, lam_2][mid]
            EBIC_min = [EBIC_l, EBIC_u, EBIC_1, EBIC_2][mid]
            continue

        if (EBIC_1 < EBIC_2) | ((EBIC_1 == EBIC_2) & (np.random.rand() > 0.5)):
            lam_upper = lam_2
            lam_2 = lam_1
            EBIC_u = EBIC_2
            EBIC_2 = EBIC_1
            h = invphi * h
            lam_1 = lam_lower + invphi2 * h

            W_1 = _MM_algorithm(C_samp, C_null, lam_1)
            pbar.update()
            EBIC_1 = _calc_EBIC(W_1, C_samp, C_null, L, beta, K_null)

            if EBIC_1 < EBIC_min:
                EBIC_min = EBIC_1
                W_best = W_1
                lam_best = lam_1

        else:
            lam_lower = lam_1
            lam_1 = lam_2
            EBIC_l = EBIC_1
            EBIC_1 = EBIC_2
            h = invphi * h
            lam_2 = lam_lower + invphi * h

            W_2 = _MM_algorithm(C_samp, C_null, lam_2)
            pbar.update()
            EBIC_2 = _calc_EBIC(W_2, C_samp, C_null, L, beta, K_null)

            if EBIC_2 < EBIC_min:
                EBIC_min = EBIC_2
                W_best = W_2
                lam_best = lam_2

    pbar.refresh()
    EBIC = EBIC_min
    return W_best, C_null, EBIC


def _calc_upper_lam(C_samp, C_null):
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
        Estimated null correlation matrix used for constructing the network.
    K_null : int
        Number of parameters for the null model. 
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
        C_null = _generate_configuration_model(np.array(C_samp), 1e-4, True)
        std_ = np.sqrt(np.diag(C_null))
        C_null = C_null / np.outer(std_, std_)
        K_null = C_samp.shape[0]
    else:
        raise ValueError(
            "Null model %s is unknown. See the Readme for the available null models."
            % null_model
        )
    return C_null, K_null


def _MM_algorithm(C_samp, C_null, lam):
    """
    Minorisation and maximisation algorithm. 
        
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
    W = _prox(C_samp - C_null, lam * Lambda)

    score_prev = -1e300
    while True:
        _W = _maximisation_step(C_samp, C_null, W, lam)
        score = _calc_penalized_loglikelihood(_W, C_samp, C_null, lam * Lambda)
        if score <= score_prev:
            break
        W = _W
        score_prev = score

    return W


def _maximisation_step(C_samp, C_null, W_base, lam):
    """
    Maximisation step of the MM algorithm. 
    (A subroutine for _MM_algorithm). 
        
    Parameters
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
    W = W_base
    Lambda = 1 / (np.power(np.abs(C_samp - C_null), 2) + 1e-20)
    inv_C_base = _fast_mat_inv_lapack(C_null + W_base)
    _diff_min = 1e300
    while (t < maxIteration) & ((t - t_best) <= maxLocalSearch) & (_diff_min > 5e-5):
        t = t + 1
        inv_C = _fast_mat_inv_lapack(C_null + W)
        gt = inv_C_base - np.matmul(np.matmul(inv_C, C_samp), inv_C)
        gt = (gt + gt.T) / 2.0
        gt = np.nan_to_num(gt)
        np.fill_diagonal(gt, 0)

        mt = b1 * mt + (1.0 - b1) * gt
        vt = b2 * vt + (1.0 - b2) * np.power(gt, 2)
        mthat = mt / (1.0 - np.power(b1, t))
        vthat = vt / (1.0 - np.power(b2, t))
        dtheta = np.divide(mthat, (np.sqrt(vthat) + eps))

        W_prev = W
        W = _prox(W - eta * dtheta, eta * lam * Lambda)
        _diff = np.max(np.abs(W - W_prev))
        np.fill_diagonal(W, 0.0)

        if _diff < _diff_min:
            _diff_min = _diff
            t_best = t

    return W


def _fast_mat_inv_lapack(Mat):
    """
    Compute the inverse of a positive semidefinite matrix.

    This function exploits the positive semidefiniteness to speed up
    the computation of matrix inversion.
        
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


def _calc_loglikelihood(W, C_samp, C_null):
    """
    Compute the log likelihood for a network. 
    
    Parameters
    ----------
    W : 2D numpy.ndarray, shape (N, N)
        Weighted adjacency matrix of a network.
    C_samp : 2D numpy.ndarray, shape (N, N)
        Sample correlation matrix. 
    C_null : 2D numpy.ndarray, shape (N, N)
        Estimated null correlation matrix used for constructing the network.

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


def _calc_penalized_loglikelihood(W, C_samp, C_null, Lambda):
    """
    Compute the penalized log likelihood for a network. 
    
    Parameters
    ----------
    W : 2D numpy.ndarray, shape (N, N)
        Weighted adjacency matrix of a network.
    C_samp : 2D numpy.ndarray, shape (N, N)
        Sample correlation matrix. 
    C_null : 2D numpy.ndarray, shape (N, N)
        Estimated null correlation matrix used for constructing the network.
    Lambda : 2D numpy.ndarray, shape (N, N)
        Lambda[i,j] is the Lasso penalty for W[i,j]. 

    Returns
    -------
    l : float
        Log likelihood for the generated network. 
    """
    return (
        _calc_loglikelihood(W, C_samp, C_null)
        - np.sum(np.multiply(Lambda, np.abs(W))) / 4
    )


def _calc_EBIC(W, C_samp, C_null, L, beta, Knull):
    """
    Compute the extended Bayesian Information Criterion (BIC) for a network. 
    
    Parameters
    ----------
    W : 2D numpy.ndarray, shape (N, N)
        Weighted adjacency matrix of a network.
    C_samp : 2D numpy.ndarray, shape (N, N)
        Sample correlation matrix.
    C_null : 2D numpy.ndarray, shape (N, N)
        Estimated null correlation matrix used for constructing the network.
    L : int
        Number of samples
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
        - 2 * L * _calc_loglikelihood(W, C_samp, C_null)
        + 4 * beta * k * np.log(W.shape[0])
    )
    return EBIC


def _prox(x, lam):
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


def _generate_configuration_model(C_samp, tolerance=1e-5):
    """
    Compute the configuration model for correlation matrices
    using the gradient descent algorithm.
    
    Parameters
    ----------
    C_samp : 2D numpy.ndarray, shape (N, N)
        Sample correlation matrix.
    tolerance: float
        Tolerance in relative error

    Returns
    -------
    C_con : 2D numpy.ndarray, shape (N, N)
        The correlation matrix under the configuration model that
        preserves the row sum (and column sum) of C_samp.
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

        K_est = np.add.outer(theta[N : 2 * N], theta[N : 2 * N]) + np.diag(theta[0:N])
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
