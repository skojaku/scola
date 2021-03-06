import numpy as np
import os
import sys
import scola


def test():
    cov = np.kron(np.eye(2) * 0.8, np.ones((5, 5)))
    cov[cov == 0] = 0.3
    np.fill_diagonal(cov, 1)
    X = np.random.multivariate_normal(np.zeros(10), cov, 300)
    L = int(X.shape[0])
    N = int(X.shape[1])
    C_samp = np.corrcoef(X.T)

    W, C_null, selected_null_model, EBIC, construct_from, all_networks = scola.corr2net.transform(C_samp, L)
    W, C_null, selected_null_model, EBIC, construct_from, all_networks = scola.corr2net.transform(C_samp, L, construct_from="pres")
    W, C_null, selected_null_model, EBIC, construct_from, all_networks = scola.corr2net.transform(C_samp, L, construct_from="auto")

test()
