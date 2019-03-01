import os
import sys
sys.path.insert(0, os.path.abspath('../'))
import scola
import numpy as np

C = np.loadtxt('motivation.txt')
W, EBIC, C_null, K_null, selected_model_id = scola.generate_network(C, 686, null_models=["hqs", "white-noise"], n_jobs = 3)

import matplotlib.pyplot as plt
plt.imshow(W)
plt.show()
