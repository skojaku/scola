.. _tutorial:

########
Tutorial
########

.. role:: python(code)
    :language: python


We demonstrate the Scola algorithm using a sample data in data/motivation.txt

Let us start from loading the Pearon correlation matrix.

>>> import numpy as np 
>>> C = np.loadtxt('data/motivation.txt') # sample data
>>> L = 686 # Number of samples 

Then, create the instance of the Scola algorithm by 

>>> import scola as sc
>>> path_to_working_directory = "result"
>>> scola = sc.Scola(path_to_working_directory)

The program will save the constructed networks in :python:`path_to_working_directory`. The directory will be created automatically if it does not exist.

Construct a network from the correlation matrix :python:`C`

>>> scola.construct_network(C, L, null_model="config")
	
The scola algorithm generates a set of networks, among which it chooses the best network in terms of the extended BIC.
All generated networks can be found in <path_to_working directory>/networks-<null_model>.pickle, e.g., result/networks-config.pickle. 

One can retrieve the best network in terms of the extended BIC by  

>>> W = scola.get_network()

where :python:`W` is the weighted adjacency matrix for the generated network.
One can obtain the null correlation matrix by 

>>> Cnull = scola.get_null_corr_matrix()

where :python:`Cnull` is the null correlation matrix. 	
