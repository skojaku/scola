========================================
A Python package for the Scola algorithm
========================================


Introduction
============

The Scola takes a correlation matrix as input and outputs a network of the variables.  
In the generated network, edges between nodes indicate correlations that are not accounted for by some trivial properties (e.g., noise and global trends affecting all variables).
The Scola stands for "Sparse network COnstruction from correlation matrices with LAsso". The details are described in our paper 


**(referenece will be here)**.


Installing Scola
================

#. You need to install ``cvxpy`` prior to install ``scola``.
   We recommend to follow `the instructions for installing cvxpy <https://ajfriendcvxpy.readthedocs.io/en/latest/install/>`_.

#. Install ``scola`` using **pip**, i.e., 

   .. code-block:: bash
   
      pip install -U scs ecos scola

#. Test the installation with pytest

   .. code-block:: bash
   
      pip install pytest && python -m pytest

   If you fail to pass the test, in many cases, ``cvxpy`` is not correctly installed (check its `dependencies <https://ajfriendcvxpy.readthedocs.io/en/latest/install/#other-platforms>`_)


Getting Started
===============

We demonstrate how to generate a network with ``scola``.

Our example data is located in `GitHub <https://raw.githubusercontent.com/skojaku/scola/master/data/sample.txt>`_, which is composed of L=300 rows and N=25 columns (space separated), where L is the number of samples and N is the number of variables (i.e., nodes). Download the data in your working directory. 

.. code-block:: python

   import numpy as np
   import pandas as pd

   X = pd.read_csv("https://raw.githubusercontent.com/skojaku/scola/master/data/sample.txt", header=None, sep=" ").values
   L = X.shape[0] # Number of samples
   N = X.shape[1] # Number of nodes

The Scola requires the sample correlation matrix and the number of samples, L.
The sample correlation matrix can be computed using numpy.  

.. code-block:: python

   # Generate NxN correlation matrix
   C_samp = np.corrcoef(X.T)

Then, provide ``C_samp`` and ``L`` to the Scola as follows.  

.. code-block:: python

   import scola
   W, EBIC, C_null, selected_model = scola.generate_network(C_samp, L)

``W`` is the weighted adjacency matrix of the generated network, where 
W[i][j] indicates the weight of the edge betwee nodes i and j.
(See `API <#module-scola.generate_network>`_ for other return values).

API
===
.. automodule:: scola.generate_network
    :members:
    :undoc-members:

Dependencies
============
``scola`` automatically installs the following packages:

* NumPy
* SciPy
* tqdm
* multiprocessing
* cvxpy

Licence
=======
GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007

.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: Table of contents:
   :glob:

   index
