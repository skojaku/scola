# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg
from scipy import sparse
from scipy import stats
import os
import sys
import tqdm
from functools import partial

from . import _scola
from . import _iscola
from . import _common
from . import search_algorithm 
from . import null_models 


def transform(C_samp, L, null_model="all", disp=True, construct_from="corr", beta = 0.5):

    """
    Generate a network from a correlation matrix
    using the Scola algorithm.

    Parameters
    ----------
    C_samp : 2D numpy.ndarray, shape (N, N)
        Sample correlation matrix. *N* is the number of nodes.
    L : int
        Number of samples.
    null_model : str or list of str or list of functions, default 'all'
        Null model to be used for constructing the network.
        The following three null models are available:
        
        - White noise model (null_model='white-noise'), 
        
        - the Hirschberger-Qi-Steuer model (null_model='hqs')

        - the configuration model (null_model='config').
        
        One can set multiple null models by a list, e.g., null_model = ["white-noise", "hqs", "config"] or equivalently null_model = "all".
        If multiple null models are given, the best one among the three null models in 
        terms of the extended Bayesian information criterion (BIC) is selected.

        To use other null models, one can set null_model = func or null_model = [func1, func2,...], where func is 
        a function taking the sample correlation matrix as the input and outputs 
        the null correlation matrix (2D numpy.ndarray, shape(N,N)), 
        the number of parameters for the null model (int), and 
        the name of the null model (str). 

    disp : bool, default True
        Set disp=True to display the progress of computation.
        Otherwise, set disp=False.

    construct_from : str, default 'corr' 
        Type of matrix to construct a network. Setting "corr" constructs based on the correlation matrix. 
        Setting "pres" constructs based on the precision matrix. If construct_from='auto', the Scola constructs networks from the correlation matrix and precision matrix. Then, it chooses the best one in terms of the extended BIC. 

    beta : float, default 0.5 
        Hyperparameter for the extended BIC. When beta = 0, the EBIC is equivalent to the BIC. The higher value yields a sparser network. Range [0,1].


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
    construct_from : str 
        construct_from='corr' or construct_from='pres' indicates that the network is constructed from 
        the correlation matrix or the precision matrix, respectively. 
    all_networks : list of dict 
        Results of all generated networks. Each dict object in the list consists of 'W', 'C_null', 'null_model', 'EBIC_min', 'construct_from' and 'W_list'. 'W_list' is a list of dict objects, in which each dict consists of a network (i.e., 'W') and its EBIC value (i.e., 'EBIC') found by the golden section search algorithm.


    Example::
        import scola
        W, C_null, selected_null_model, EBIC, construct_from, all_networks = scola.corr2net.transform(C_samp, L)
    """

    if type(C_samp) is not np.ndarray:
        raise TypeError("C_samp must be a numpy.ndarray")

    if (type(L) is not int) and (type(L) is not float):
        raise TypeError("L must be an integer")

    if type(disp) is not bool:
        raise TypeError("disp must be a bool")

    if not ( (type(null_model) is str) or (type(null_model) is list) ):
        raise TypeError("null_model must be a string, a list of strings or a list of functions")

    def _to_null_model_func(n):
        if isinstance(n, str):
            if n == "white-noise":
                return null_models.white_noise_model
            elif n == "hqs":
                return null_models.hqs_model
            elif n == "config":
                return null_models.configuration_model
        elif hasattr(n, '__call__'):
            return n

    if null_model == "all": 
        null_model = ["white-noise", "hqs", "config"]

    if isinstance(null_model, list):
        _null_models = [_to_null_model_func(n) for n in null_model]
    elif isinstance(null_model, str):
        _null_models = [_to_null_model_func(null_model)]

    if construct_from=='auto':
        mat_types = ["corr", "pres"]
    else:
        mat_types = [construct_from]

    if not ((type(beta) is float) or (type(beta) is int)):
        raise TypeError("beta must be float")

    if (beta < 0) or (1 < beta):
        raise TypeError("beta must be in range [0,1]")

    # Remove inactive nodes
    active_nodes = np.where(np.abs(np.sum(C_samp, axis=1)) > 0)[0]
    R = sparse.csc_matrix(
        (np.ones(len(active_nodes)), (list(range(len(active_nodes))), active_nodes)),
        shape=(len(active_nodes), C_samp.shape[0]),
    ).toarray()
    C_samp = np.matmul(np.matmul(R, C_samp), R.T)

    # Improve the degeneracy of C_samp for the computational stability if it has too small eigenvalues
    C_samp = _common._remedy_degeneracy(C_samp, rho = 1e-6)

    # pbar is used for computing and displaying the progress of computation.
    pbar = tqdm.tqdm(
        disable=(disp is False), total=(13 * len(_null_models) * len(mat_types))
        #disable=(disp is False), total=(13 * len(_null_models) * len(mat_types))
    )

    res = []
    for null_model in _null_models:

        # Estimate null correlation matrix
        C_null, K_null, null_model_name = null_model(C_samp)

        iC_null = np.linalg.pinv(C_null)

        for mat_type in mat_types:

            if mat_type == "corr":

                estimator = _scola.Scola()

                W, C_null, EBIC_min, all_networks = search_algorithm.golden_section_search(
                    C_samp, L, C_null, K_null, estimator, beta, pbar, disp
                )
                W = np.matmul(np.matmul(R.T, W), R)

                res += [
                    {
                        "W": W,
                        "C_null": C_null,
                        "null_model": null_model_name,
                        "EBIC_min": EBIC_min,
                        "construct_from": mat_type,
                        "W_list": all_networks
                    }
                ]

            elif mat_type == "pres":

                estimator = _iscola.iScola()
                W, C_null, EBIC_min, all_networks = search_algorithm.golden_section_search(
                    C_samp, L, iC_null, K_null, estimator, beta, pbar, disp
                )
                W = np.matmul(np.matmul(R.T, W), R)


                res += [
                    {
                        "W": W,
                        "C_null": iC_null,
                        "null_model": null_model_name,
                        "EBIC_min": EBIC_min,
                        "construct_from": mat_type,
                        "W_list": all_networks
                    }
                ]

    pbar.close()

    idx = np.argmin(np.array([r["EBIC_min"] for r in res]))
    return res[idx]["W"], res[idx]["C_null"], res[idx]["null_model"], res[idx]["EBIC_min"], res[idx]["construct_from"], res
