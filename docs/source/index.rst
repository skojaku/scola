========================================
A Python package for the Scola algorithm
========================================

.. image:: https://travis-ci.org/skojaku/scola.svg?branch=master

Introduction
============

The Scola is an algorithm that takes a correlation matrix as input and outputs a network of the variables. 
In the generated network, edges between nodes indicate correlations that are not accounted for by some trivial properties (e.g., noise and global trends affecting all variables).
Please cite the paper if you use this algorithm: 


**(referenece will be here)**.

Installing Scola
================
``scola`` is a pure Python package that supports both Python 2.x and 3.x.
We test ``scola`` on Ubuntu, CentOS, macOS and Windows.

Install with pip
----------------
.. code-block:: bash

   pip install scola

Install from source
-------------------
Download `the source file <https://github.com/skojaku/scola>`_ from GitHub.
Then, under /scola directory, type

.. code-block:: bash

   python setup.py install 

Getting Started
===============

We demonstrate how to generate a network with ``scola``.

Our example data is located in `GitHub <https://raw.githubusercontent.com/skojaku/scola/master/data/sample.txt>`_, which is composed of L=300 rows and N=25 columns (space separated), where L is the number of samples and N is the number of variables (i.e., nodes). 
The code below loads the data. 

.. code-block:: python

   import numpy as np
   import pandas as pd

   X = pd.read_csv("https://raw.githubusercontent.com/skojaku/scola/master/data/sample.txt", header=None, sep=" ").values
   L = X.shape[0] # Number of samples
   N = X.shape[1] # Number of nodes

Then, compute the sample correlation matrix by 

.. code-block:: python

   C_samp = np.corrcoef(X.T) # NxN correlation matrix

Finally, provide ``C_samp`` and ``L`` to the Scola as follows. 

.. code-block:: python

   import scola
   W, EBIC, C_null, selected_model = scola.generate_network(C_samp, L)

``W`` is the weighted adjacency matrix of the generated network, where 
W[i][j] indicates the weight of the edge betwee nodes i and j.
See `API <#module-scola.generate_network>`_ for other return values.

API
===
.. automodule:: scola.generate_network
    :members:
    :undoc-members:

Dependencies
============
``scola`` has dependencies with the following packages:

* NumPy
* SciPy
* tqdm

Licence
=======
GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007

.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: Table of contents:
   :glob:

   index
