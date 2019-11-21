import numpy as np
import pandas as pd

X = pd.read_csv("https://raw.githubusercontent.com/skojaku/scola/master/data/sample.txt", header=None, sep=" ").values
L = X.shape[0] # Number of samples
N = X.shape[1] # Number of nodes

C_samp = np.corrcoef(X.T) # NxN correlation matrix

import scola

#W, C_null, selected_null_model, EBIC, input_mat_type, all_networks = scola.corr2net.transform(C_samp, L)
C_null, K_null, name = scola.null_models.configuration_model(C_samp)

print(C_null)
