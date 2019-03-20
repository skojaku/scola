========================================
A Python package for the Scola algorithm
========================================

.. image:: https://travis-ci.org/skojaku/scola.svg?branch=master

Introduction
============

The Scola is an algorithm that takes a correlation matrix as input and outputs a network. 
In the generated network, edges between nodes indicate correlations that are not accounted for by some expected properties (e.g., noise independent for different variables or a global trend).
Please cite the paper if you use this package: 


**(referenece will be here)**.

Installing the Scola
================
``scola`` supports both Python 2.x and 3.x and can be installed on Ubuntu, CentOS, macOS and Windows.
We recommend using **pip** for installation. 

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

An example data, `sample.txt <https://raw.githubusercontent.com/skojaku/scola/master/data/sample.txt>`_, is available on  `GitHub <https://github.com/skojaku/scola>`_, which is composed of L=300 rows and N=25 columns (space separated), where L is the number of samples and N is the number of variables (i.e., nodes). 
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

Finally, provide ``C_samp`` and ``L`` to estimate the network and associated null model: 

.. code-block:: python

   import scola
   W, EBIC, C_null, selected_model = scola.generate_network(C_samp, L)

``W`` is the weighted adjacency matrix of the generated network, where 
W[i,j] indicates the weight of the edge between nodes i and j.
See `API <#module-scola.generate_network>`_ for other return values.

API
===
.. automodule:: scola.generate_network
    :members:
    :undoc-members:

Dependencies
============
``scola`` has dependencies on the following packages:

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
