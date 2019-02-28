############
Installation
############

.. role:: python(code)
    :language: python

Requirements
------------

``scola`` is compatible with python 3.4 or newer. 
We recommend using pip for installation. 
If you prefer Anaconda or other package managers, you can instead install ``scola`` from source. 

We have tested ``scola`` on Ubuntu and CentOS.

.. todo:: Check compatibility to Mac and Windows when I have time to do it


Install with pip
----------------

#. `Upgrade pip to the latest <https://pip.pypa.io/en/stable/installing/>`_

#. Install ``scola``  with pip.

   .. code-block:: bash

        pip3 install scola

   If you don't have a root access, try :python:`--user` flag:

   .. code-block:: bash

        pip3 install --user scola


Install from source
-------------------

#. Download `the source <https://github.com/skojaku/scola/>`_ from GitHub.

#. Move to the top directory, i.e., /scola

#. Type

   .. code-block:: bash

    python setup.py install
    

