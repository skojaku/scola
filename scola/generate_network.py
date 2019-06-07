# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg
from scipy import sparse
from scipy import stats
import os
import sys
import tqdm
from functools import partial
from ._scola import *
from ._iscola import *
from ._common import * 
from ._golden_section_search import _golden_section_search
from ._compute_null_correlation_matrix import _compute_null_correlation_matrix 

def generate_network(C_samp, L, null_model="all", disp=True, input_matrix="all"):
    """
    Generate a network from a correlation matrix
    using the Scola algorithm.

    Parameters
    ----------
    C_samp : 2D numpy.ndarray, shape (N, N)
        Sample correlation matrix. *N* is the number of nodes.
    L : int
        Number of samples.
    null_model : str, default 'all'
        Null model to be used for constructing the network.
        One can use the white noise model 
        (null_model='white-noise'), the Hirschberger-Qi-Steuer 
        model (null_model='hqs') or the configuration model (null_model='config'). 
        If null_model='all', then the best one among the three null models in 
        terms of the extended Bayesian information criterion (BIC) is selected.
    disp : bool, default True
        Set disp=True to display the progress of computation.
        Otherwise, set disp=False.
    input_matrix : string, default "all" 
        Matrix to be used to construct a network.
            - Setting "corr" constructs based on the correlation matrix 
            - Setting "pres" constructs based on the precision matrix

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

    if input_matrix == "all":
        mat_types = ["corr", "pres"]
    else:
        mat_types = [input_matrix]

    # pbar is used for computing and displaying the progress of computation.
    pbar = tqdm.tqdm(disable=(disp is False), total=(13 * len(_null_models) * len(mat_types)))
    res = []
    for null_model in _null_models:

        # Estimate null correlation matrix
        C_null, K_null = _compute_null_correlation_matrix(C_samp, null_model)

        for mat_type in mat_types:
    
            if mat_type == "corr":
                estimator = Scola()
            elif mat_type == "pres":
                estimator = iScola()
    
            W, C_null, EBIC_min = _golden_section_search(
                C_samp, L, C_null, K_null, estimator, 0.5, pbar, disp
            )
            res += [{"W":W, "C_null":C_null, "null_model":null_model, "EBIC_min":EBIC_min, "mat_type":mat_type}]
    pbar.close()

    idx = np.argmin(np.array([r["EBIC_min"] for r in res]))
        
    return res[idx], res


