# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg
from scipy import sparse
from scipy import stats
import os
import tqdm
import sys
from functools import partial
from ._scola import *
from ._common import _fast_mat_inv_lapack
from ._common import _comp_EBIC 

def _golden_section_search(C_samp, L, C_null, K_null, estimator, beta, pbar, disp):
    """
    Find the Lasso penalty that minimises the extended BIC using
    the golden-section search method.

    Parameters
    ----------
    C_samp : 2D numpy.ndarray, shape (N, N)
        Sample correlation matrix.
    L : int
        Number of samples.
    null_model : str
        Name of the null model.
    beta : float
        Hyperparameter for the extended BIC.
    pbar : tqdm instance
        This instance is used for computing and displaying 
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

    lam_upper = estimator.comp_upper_lam(C_samp, C_null)
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
            W_u = estimator.detect(C_samp, C_null, lam_upper)
            pbar.update()
            W_1 = estimator.detect(C_samp, C_null, lam_1)
            pbar.update()
            W_2 = estimator.detect(C_samp, C_null, lam_2)
            pbar.update()

            EBIC_l = _comp_EBIC(W_l, C_samp, C_null, L, beta, K_null)
            EBIC_u = _comp_EBIC(W_u, C_samp, C_null, L, beta, K_null)
            EBIC_1 = _comp_EBIC(W_1, C_samp, C_null, L, beta, K_null)
            EBIC_2 = _comp_EBIC(W_2, C_samp, C_null, L, beta, K_null)

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

            W_1 = estimator.detect(C_samp, C_null, lam_1)
            pbar.update()
            EBIC_1 = _comp_EBIC(W_1, C_samp, C_null, L, beta, K_null)

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

            W_2 =  estimator.detect(C_samp, C_null, lam_2)
            pbar.update()
            EBIC_2 = _comp_EBIC(W_2, C_samp, C_null, L, beta, K_null)

            if EBIC_2 < EBIC_min:
                EBIC_min = EBIC_2
                W_best = W_2
                lam_best = lam_2

    pbar.refresh()
    EBIC = EBIC_min
    return W_best, C_null, EBIC




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
