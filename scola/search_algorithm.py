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



def golden_section_search(C_samp, L, C_null, K_null, estimator, beta, pbar, disp):

    lam_upper = estimator.comp_upper_lam(C_samp, C_null)
    lam_lower = 0

    W_best, C_null, EBIC, lam_best, all_networks = _golden_section_search(C_samp, L, C_null, K_null, estimator, beta, lam_lower, lam_upper, pbar, False, disp)

    return W_best, C_null, EBIC, all_networks


def _golden_section_search(C_samp, L, C_null, K_null, estimator, beta, lam_lower, lam_upper, pbar, W_interpolate, disp):
    """
    Find the Lasso penalty that minimises the extended BIC using
    the golden-section search method.

    Parameters
    ----------
    C_samp : 2D numpy.ndarray, shape (N, N)
        Sample correlation matrix.
    L : int
        Number of samples.
    C_null : str
       Null model (set correlation matrix for scola and precision matrix for iscola) 
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
    all_networks : list of dict 
        Results of all generated networks. Each dict object in the list consists of 'W' and 'EBIC'.
    """

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
    
    all_networks = []
    for k in range(n):
        if k == 0:

            #W_l = C_samp - C_null
            W_l = estimator.detect(C_samp, C_null, lam_lower)
            pbar.update()
            W_u = estimator.detect(C_samp, C_null, lam_upper)
            pbar.update()
            W_1 = estimator.detect(C_samp, C_null, lam_1)
            pbar.update()
            W_2 = estimator.detect(C_samp, C_null, lam_2)
            pbar.update()

            EBIC_l = _comp_EBIC(
                W_l, C_samp, C_null, L, beta, K_null, estimator.input_matrix_type
            )
            EBIC_u = _comp_EBIC(
                W_u, C_samp, C_null, L, beta, K_null, estimator.input_matrix_type
            )
            EBIC_1 = _comp_EBIC(
                W_1, C_samp, C_null, L, beta, K_null, estimator.input_matrix_type
            )
            EBIC_2 = _comp_EBIC(
                W_2, C_samp, C_null, L, beta, K_null, estimator.input_matrix_type
            )

            mid = np.argmin([EBIC_l, EBIC_u, EBIC_1, EBIC_2])
            W_best = [W_l, W_u, W_1, W_2][mid]
            lam_best = [lam_lower, lam_upper, lam_1, lam_2][mid]
            EBIC_min = [EBIC_l, EBIC_u, EBIC_1, EBIC_2][mid]

            all_networks += [
                {
                    "W": W_l,
                    "EBIC": EBIC_l,
                },
                {
                    "W": W_u,
                    "EBIC": EBIC_u,
                },
                {
                    "W": W_1,
                    "EBIC": EBIC_1,
                },
                {
                    "W": W_2,
                    "EBIC": EBIC_2,
                }
            ]
            continue

        if (EBIC_1 < EBIC_2) | ((EBIC_1 == EBIC_2) & (np.random.rand() > 0.5)):
            lam_upper = lam_2
            lam_2 = lam_1
            EBIC_u = EBIC_2
            EBIC_2 = EBIC_1
            h = invphi * h
            lam_1 = lam_lower + invphi2 * h

            if W_interpolate == True:
                W_1 = estimator.detect(C_samp, C_null, lam_1, (W_l + W_1)/2)
            else:
                W_1 = estimator.detect(C_samp, C_null, lam_1)


            pbar.update()
            EBIC_1 = _comp_EBIC(
                W_1, C_samp, C_null, L, beta, K_null, estimator.input_matrix_type
            )

            if EBIC_1 < EBIC_min:
                EBIC_min = EBIC_1
                W_best = W_1
                lam_best = lam_1

            all_networks += [
                {
                    "W": W_1,
                    "EBIC": EBIC_1
                }
            ]

        else:
            lam_lower = lam_1
            lam_1 = lam_2
            EBIC_l = EBIC_1
            EBIC_1 = EBIC_2
            h = invphi * h
            lam_2 = lam_lower + invphi * h

            if W_interpolate == True:
                W_2 = estimator.detect(C_samp, C_null, lam_2, (W_2 + W_u)/2)
            else:
                W_2 = estimator.detect(C_samp, C_null, lam_2)

            pbar.update()
            EBIC_2 = _comp_EBIC(
                W_2, C_samp, C_null, L, beta, K_null, estimator.input_matrix_type
            )

            if EBIC_2 < EBIC_min:
                EBIC_min = EBIC_2
                W_best = W_2
                lam_best = lam_2

            all_networks += [
                {
                    "W": W_2,
                    "EBIC": EBIC_2
                }
            ]

    pbar.refresh()
    EBIC = EBIC_min
    return W_best, C_null, EBIC, lam_best, all_networks
