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
    W, EBIC, C_null, selected_model = scola.generate_network(C_samp, L)


def test_2():
    cov = np.kron(np.eye(2) * 0.8, np.ones((5, 5)))
    cov[cov == 0] = 0.3
    np.fill_diagonal(cov, 1)
    X = np.random.multivariate_normal(np.zeros(10), cov, 300)
    L = int(X.shape[0])
    N = int(X.shape[1])
    C_samp = np.corrcoef(X.T)
    Crand = scola._estimate_configuration_model(C_samp)
    relative_error = np.max(np.abs(np.sum(C_samp - Crand, 1)) / np.sum(C_samp, 1))
    assert relative_error <= 1e-5
