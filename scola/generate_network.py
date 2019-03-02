# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg
from scipy import sparse
from scipy import stats
import os
import tqdm
import sys
#import configcorr as cg
from functools import partial
import cvxpy as cv
from multiprocessing import Pool, current_process

def generate_network(C_samp, L, null_model='all', disp=True, n_jobs = 3):
    """
    Generate a network from a correlation matrix
    using the Scola algorithm

    Parameters
    ----------
    C_samp : 2D numpy.matrix, shape (N, N)
        Sample correlation matrix. N is the number of nodes.
    L : int
        Number of samples
    null_model : str, default 'all' 
        Name of the null model to be used for constructing the network.  
        Available null models are
        the white noise model (null_model='white-noise'),
        the Hirschberger-Qu-Steuer model (null_model='hqs')
        and the configuration model (null_model='config')
        If null_model='all', then the Scola selects the best one among the three null models in terms of the extended BIC.  
    disp : bool, default True
        Set disp=True to disply the progress.
        Otherwise set disp=False.

    n_jobs : int, default 3
        Number of jobs to run in parallel.

    Returns
    -------
    W : 2D numpy.matrix, shape (N, N)
        Weighted adjacency matrix of the generated network.
    C_null : 2D numpy.matrix, shape (N, N)
        Null correlation matrix.
    null_model : str
        Name of the null model.
    EBIC : float
        The extended BIC for the generated network.
    """

    # Check input types
    if type(C_samp) is np.ndarray:
        C_samp = np.matrix(C_samp)

    if type(C_samp) is not np.matrix:
        raise TypeError("C_samp must be a numpy.matrix")

    if (type(L) is not int) and (type(L) is not float):
        raise TypeError("L must be an integer")

    if type(disp) is not bool:
        raise TypeError("disp must be a bool")

    if type(null_model) is not str:
        raise TypeError("null_model must be a string")
  
    if null_model == 'all':
        null_models_to_be_used = ["white-noise", "hqs", "config"]
        pool = Pool(n_jobs)
        res = pool.map(partial(_gen_net_, C_samp, L, disp=disp, gamma=0.5), [(null_model, i) for i, null_model in enumerate(null_models_to_be_used)])
        pool.close()
        pool.join()
        idx = np.argmin(np.array([r[2] for r in res]))
        return res[idx]
    else:
        return _gen_net_(C_samp, L, null_model_process_id = [null_model, 0], disp=disp, gamma=gamma)

def _gen_net_(C_samp, L, null_model_process_id, disp, gamma):
    
    null_model = null_model_process_id[0]
    null_model_id = null_model_process_id[1]
    
    lam_lower = 0 
    lam_upper = 1.0

    invphi = (np.sqrt(5) - 1) / 2
    invphi2 = (3 - np.sqrt(5)) / 2
    h = lam_upper-lam_lower
    lam_1 = lam_lower + invphi2 * h
    lam_2 = lam_lower + invphi * h
    n = int(np.ceil(np.log(0.01/h)/np.log(invphi)))
    N = C_samp.shape[0]
    W_best = []
    lam_best = 0 
    EBIC_min = 0 

    pbar = tqdm.tqdm(disable=(disp is False), position=null_model_id, total=n+2, desc=null_model)
    pbar.set_description("Estimating null correlation matrix (%s)" % null_model)
    pbar.update()
    if type(null_model) is str:
        C_null, K_null = _compute_null_correlation_matrix(C_samp, null_model, False)
    else:
        null_model = "user-defined"
        C_null = null_model[0]
        K_null = null_model[1]
    
    pbar.set_description("Constructing networks (%s)" % null_model)
    pbar.update()
    
    for k in range(n):
        if k == 0:

            W_l = C_samp - C_null
            W_l = _MM_algorithm(C_samp, C_null, lam_lower)
            W_u = _MM_algorithm(C_samp, C_null, lam_upper)
            W_1 = _MM_algorithm(C_samp, C_null, lam_1)
            W_2 = _MM_algorithm(C_samp, C_null, lam_2)
            
            EBIC_l = _calc_EBIC(
                    W_l, C_samp, C_null, L, gamma, K_null)
            EBIC_u = _calc_EBIC(
                    W_u, C_samp, C_null, L, gamma, K_null)
            EBIC_1 = _calc_EBIC(
                    W_1, C_samp, C_null, L, gamma, K_null)
            EBIC_2 = _calc_EBIC(
                    W_2, C_samp, C_null, L, gamma, K_null)
       
            mid = np.argmin([EBIC_l, EBIC_u, EBIC_1, EBIC_2])
            W_best = [W_l, W_u, W_1, W_2][mid]
            lam_best = [lam_lower, lam_upper, lam_1, lam_2][mid]
            EBIC_min = [EBIC_l, EBIC_u, EBIC_1, EBIC_2][mid]
            pbar.update()
            continue

        if (EBIC_1 < EBIC_2) |\
                ((EBIC_1 == EBIC_2) & (np.random.rand() > 0.5)):

            lam_upper = lam_2
            lam_2 = lam_1
            EBIC_u = EBIC_2
            EBIC_2 = EBIC_1
            h = invphi * h
            lam_1 = lam_lower + invphi2 * h

            W_1 = _MM_algorithm(C_samp, C_null, lam_1)

            EBIC_1 = _calc_EBIC(
                    W_1, C_samp, C_null, L, gamma, K_null)
            
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
            EBIC_2 = _calc_EBIC(
                    W_2, C_samp, C_null, L, gamma, K_null)

            if EBIC_2 < EBIC_min:
                    EBIC_min = EBIC_2
                    W_best = W_2
                    lam_best = lam_2

        pbar.update()
    pbar.close()
    return W_best, C_null, EBIC_min, null_model

def _generate_configuration_model(C, tolerance = 1e-5, transform_to_corr_mat = True):

    if transform_to_corr_mat == True:
        cov = np.asanyarray(C)
        std_ = np.sqrt(np.diag(cov))
        _C = cov / np.outer(std_, std_)
    else:
        _C = C

    N = _C.shape[0]

    C_con = cv.Variable((N, N), PSD=True) 

    objective = cv.Minimize(-cv.log_det(C_con))  

    constraints = [cv.sum(C_con, axis=0) == _C.sum(axis = 0), cv.diag(C_con) == np.diag(_C)]

    prob = cv.Problem(objective, constraints)
    prob.solve(verbose = False, eps = tolerance)

    return C_con.value

def _compute_null_correlation_matrix(C_samp, null_model, disp):
    C_null = []
    K_null = -1

    if null_model == "white-noise":
        C_null = np.matrix(np.eye(C_samp.shape[0]))
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
            'Null model %s is unknown. See the Readme for the available null models' % null_model)
    return C_null, K_null

def _MM_algorithm(C_samp, C_null, lam):
    N = C_samp.shape[0]
    Lambda = 1/(np.power(np.abs(C_samp-C_null), 2)+1e-20)
    W = _prox(C_samp - C_null, lam * Lambda)
     
    score_prev = -1.7976931348623157e+308
    while True:
        _W = _maximisation_step(C_samp, C_null, W, lam)
        score = _penalized_likelihood(_W, C_samp, C_null, lam, Lambda)
        if score <= score_prev:
            break
        W = _W
        score_prev = score

    return W

def _maximisation_step(C_samp, C_null, W_base, lam):
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

    Lambda = 1/(np.power(np.abs(C_samp-C_null), 2)+1e-20)

    inv_C_base = _fast_inv_mat_lapack(C_null + W_base)
    while (t < maxIteration) & (t <= (t_best + maxLocalSearch+1)):
        t = t + 1
        inv_C = _fast_inv_mat_lapack(C_null+W)
        gt = inv_C_base - inv_C @ C_samp@ inv_C
        gt = (gt + gt.T)/2
        gt = np.nan_to_num(gt)
        np.fill_diagonal(gt, 0)
        mt = b1 * mt + (1-b1)*gt
        vt = b2 * vt + (1-b2)*np.power(gt, 2)
        mthat = mt / (1-np.power(b1, t))
        vthat = vt / (1-np.power(b2, t))
        dtheta = np.divide(mthat, (np.sqrt(vthat) + eps))
        
        W = _prox(W - eta * dtheta, eta * lam * Lambda)
        np.fill_diagonal(W, 0)

        if (t % quality_assessment_interval) == 0:
            s = _penalized_likelihood(W, C_samp, C_null, lam, Lambda)
            if s > maxscore:
                W_best = W
                maxscore = s
                t_best = t

    return W_best

def _fast_inv_mat_lapack(M):
    zz, _ = linalg.lapack.dpotrf(M, False, False)
    inv_M, info = linalg.lapack.dpotri(zz)
    inv_M = np.triu(inv_M) + np.triu(inv_M, k=1).T
    return inv_M

def _loglikelihood(W, C_samp, C_null):
    Cov = W + C_null
    w, v = np.linalg.eig(Cov)
    if np.min(w) <0:
        v = v[w>0]
        w = w[w>0]
    iCov = np.real(v @ np.diag(1/w) @ v.T)
    l = -0.5 * np.sum(np.log(w)) - 0.5 * np.trace(C_samp @ iCov)\
            - 0.5 * Cov.shape[0] * np.log(2 * np.pi)
    return np.real(l)

def _calc_EBIC(W, C_samp, C_null, L, gamma, Knull):
    k = Knull + np.count_nonzero(W)/2
    EBIC = np.log(L) * k - 2 * L * _loglikelihood(W, C_samp, C_null) + 4 * gamma * k * np.log(W.shape[0])
    return EBIC

def _prox(y, lam):
    return np.multiply(np.sign(y),
            np.maximum(np.abs(y)
                - lam, np.matrix(np.zeros(y.shape))))

def _penalized_likelihood(W, C_samp, C_null, lam, Lambda):
    return _loglikelihood(W, C_samp, C_null)\
            - lam * np.sum(np.dot(Lambda, np.abs(W)))/2
            #- lam * np.sum(np.dot(Lambda, np.abs(W)))/2
