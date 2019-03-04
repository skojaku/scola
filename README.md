About this library
==================

A Python module for the Scola algorithm for constructing networks from correlation matrices.
 
Please cite:

`Sadamori Kojaku and Naoki Masuda. xxx Preprint arXiv: xxx. 

Contents
========
Python module:
  - scola/__init__.py
  - scola/Scola.py

Example data and code:
  - example/example.py
  - example/motivation.txt

Others (for PyPi registration and Travis-CI):
  - MANIFEST.in
  - requirements.txt
  - setup.py
  - .travis.yml
  - tests

Installation
============

To install, type
      
.. code-block:: bash

  pip3 install scola 

If you don't have root privilege, use -user flag, i.e.,  
      
.. code-block:: bash

  pip3 install --user scola 


Usage
=====

See the document. 

Requirements
============
- Python 3.4 or later
- Numpy 1.14 or later
- SciPy 1.1 or later
- NetworkX 2.0 or later
- pybind11 2.2 or later 
An algorithm for constructing networks from correlation matrices with Lasso


# core-periphery-detection
[![Build Status](https://travis-ci.org/skojaku/scola.svg?branch=master)](https://travis-ci.org/skojaku/scola)


See [docs](https://scola.readthedocs.io/en/latest/) for installation, tutorials, and available algorithms.
 
