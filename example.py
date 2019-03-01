import numpy as np 
import matplotlib.pyplot as plt

#cov =  np.kron(np.eye(5) * 0.8, np.ones((5,5)))
#cov[cov==0] = 0.3
#np.fill_diagonal(cov, 1)
#X = np.random.multivariate_normal(np.zeros(25), cov, 300)


X = np.loadtxt('data/sample.txt')
C_samp = np.corrcoef(X.T)
L = X.shape[0]


import scola
W, C_null, EBIC, model_selected = scola.generate_network(C_samp, L, null_models=['white-noise'])

#fig, axes = plt.subplots(nrows=1, ncols=2)
#axes[0].imshow(C_samp, label="sample")
#axes[0].set_title('C_samp')
#im = axes[1].imshow(W)
#axes[1].set_title('W')

#fig.subplots_adjust(right=0.8)
#fig.colorbar(im, cax=fig.add_axes([0.85, 0.15, 0.05, 0.7]))
#plt.show()
