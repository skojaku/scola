================================================
Constructing networks with different null models
================================================

With the Scola, two nodes are adjacent in the network if and only if the correlation between the nodes is considerably different from that in a null model.
In this example, we will show that diffferent null models will generate considerably different netwowkrs.

We use historic S&P 500 stock price data retrieved from Yahoo finance, `sample.txt <https://raw.githubusercontent.com/skojaku/scola/master/data/sample.txt>`_, is available on  `GitHub <https://raw.githubusercontent.com/skojaku/scola/master/data/sp500-log-return.csv>`_, which consists of the log return of the price of N=xxx stock prices between xxxx and xxx (L=xxx days). 
The code to retrieve the data can be found in `here <https://raw.githubusercontent.com/skojaku/scola/master/data/get_sp500_stock_prices.py>`.  
An example data, , which is composed of xx=300 rows and N=25 columns (space separated), where L is the number of samples and N is the number of variables (i.e., nodes). 
The following code loads the data. 

.. code-block:: python

   import numpy as np
   import pandas as pd

   X = pd.read_csv("https://raw.githubusercontent.com/skojaku/scola/master/data/sample.txt", header=None, sep=" ").values
   L = X.shape[0] # Number of samples
   N = X.shape[1] # Number of nodes

Then, compute the sample correlation matrix by 

.. code-block:: python

   C_samp = np.corrcoef(X.T) # NxN correlation matrix

``C_samp`` looks like

.. figure:: fig/C\_samp.png
   :scale: 20 %
   :align: center 


Finally, provide ``C_samp`` and ``L`` to estimate the network and associated null model: 

.. code-block:: python

   import scola
   W, C_null, selected_null_model, EBIC, construct_from, all_networks = scola.corr2net.transform(C_samp, L)

``W`` is the weighted adjacency matrix of the generated network, where 
W[i,j] indicates the weight of the edge between nodes i and j.

The ``W`` looks like

.. figure:: fig/W.png
   :scale: 20 %
   :align: center 

See the :ref:`scola_package` for other return values.

Scola can construct a network from precision matrices, which is often different from that constructed from correlation matrices. 
To do this, give an extra parameter ``construct_from='pres'``: 

.. code-block:: python

   import scola
   W, C_null, selected_null_model, EBIC, construct_from, all_networks = scola.corr2net.transform(C_samp, L, construct_from="pres")

which produces a different network:

.. figure:: fig/Wpres.png
   :scale: 20 %
   :align: center 

If one sets ``construct_from='auto'``, the Scola constructs networks from correlation matrices and precision matrices. 
Then, it chooses the one that best represents the given data in terms of the extended BIC.
The selected type of the matrix is indicated by ``construct_from`` in the return variables. 
