#######
Example
#######

.. role:: python(code)
    :language: python

.. todo:: Consider which data would be appropriate again.


This example demonstrates how to generate a network with the Scola.
We assume that the current directory is /scola.


We use data located in data/sample.txt, which is composed of L rows and N columns, where L is the number of samples and N is the number of variables (i.e., nodes).
To construct a network, we compute the correlation matrix of nodes. 

.. code-block:: python

   import numpy as np
   
   # Load data 
   X = np.loadtxt('data/sample.txt') # numpy.matrix of shape (L, N)
   L = X.shape[0] # Number of samples
   N = X.shape[1] # Number of nodes

   # Generate NxN correlation matrix
   C_samp = np.corrcoef(X.T)

Then, we provide ``C_samp`` and ``L`` to the Scola.

.. code-block:: python

   import scola

   # Generate a network from C_samp 
   W, EBIC, C_null, selected_model_id = scola.generate_network(C_samp, L)


``W`` is the weighted adjacency matrix of the generated network. 
(See ??? for other return values).

To see what ``C_samp`` and ``W`` are, visualize them by  

.. code-block:: python

   import matplotlib.pyplot

   fig, axes = plt.subplots(nrows=1, ncols=2)
   axes[0].imshow(C_samp)
   im = axes[1].imshow(W)
   axes[1].set_title('W')
   axes[0].set_title('C_samp')
   fig.subplots_adjust(right=0.8)
   fig.colorbar(im, cax=fig.add_axes([0.85, 0.15, 0.05, 0.7]))
   plt.show()

Visual inspection suggests that ``C_samp`` contains five groups of strongly correlated nodes.
The nodes in different groups are weakly correlated with each other (correlation value 0.3 on average).
``W`` contains five groups of densely interconnected nodes, akin to ``C_samp``.
However, nodes in different groups are not adjacent with each other, meaning that Scola regarded the inter-group correlations as spurious and thus did not place edges.


The Scola places edges between nodes if the correlation between them are not accounted for by a null model.  
Different null models may yield different networks. 
By default, the Scola automatically selectes the most appropriate null model among the following three null models;
    * White noise model ('white-noise')
    * Hirschberger-Qu-Steuer model ('hqs')
    * Configuration model ('config')

One can specify the null model by giving a list of names, e.g.,  

.. code-block:: python

   # Generate network 
   W, EBIC, C_null, selected_model_id = scola.generate_network(C, L, null_model=["white-noise"]) # Generate a network

Alternatively, one can use a user-defined null model by setting a tuple (C_null, K_null), where

    * **C_null** : (*2D numpy.matrix, shape(N,N)*) - Correlation matrix under the null model.
    * **K_null** : (*int*) - Number of parameters of C_null.

.. code-block:: python

   # Generate network 
   W, EBIC, C_null, selected_model_id = scola.generate_network(C, L, null_model=["white-noise"]) # Generate a network

