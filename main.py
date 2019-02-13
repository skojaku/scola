import scola
import numpy as np

C = np.loadtxt('motivation.txt')
sc = scola.Scola('test')
W, EBIC, lam = sc.run(C, 100, null_model="config")
W = sc.get_network()
C_null = sc.get_null_corr_matrix()
C = sc.get_corr_matrix()

import matplotlib.pyplot as plt
plt.imshow(W)
plt.show()
