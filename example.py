import numpy as np 
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

X = pd.read_csv("https://raw.githubusercontent.com/skojaku/scola/master/data/sample.txt", header=None, sep=" ").values
L = X.shape[0] # Number of samples
N = X.shape[1] # Number of nodes
# Generate NxN correlation matrix
C_samp = np.corrcoef(X.T)
import scola
W, EBIC, C_null, selected_model = scola.generate_network(C_samp, L)

print(type(W), type(C_samp), selected_model)

plt.imshow(W)
plt.show()
