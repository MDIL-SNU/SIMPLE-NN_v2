.. _install:

============
Installation
============

------------
Requirements
------------
- Python ``3.6-3.9``
- PyTorch ``1.5.0-1.10.1``
- LAMMPS ``29Oct2020`` or later

----

----------
Procedures
----------

1. Pytorch
----------
Install PyTorch: https://pytorch.org/get-started/locally

Choose the PyTorch of stable release for ``Python``. If you have CUDA-capable system, please download PyTorch with CUDA that makes training much faster.

To check if your GPU driver and CUDA are enabled by PyTorch, run the following commands in python to return whether or not the CUDA driver is enabled: 

.. code-block:: python3

    import torch.cuda
    torch.cuda.is_available()

2. SIMPLE-NN
------------

2-1. Download from github url
=============================
If you have git, you can download a SIMPLE-NN through cloning from repository. This will create SIMPLE-NN directory.

.. code-block:: bash

    git clone https://github.com/MDIL-SNU/SIMPLE-NN_v2.git SIMPLE-NN

2-2. Download as a zip file
===========================
You can download a current SIMPLE-NN source package from link below. 
Click the green ``Code`` button on upper right side and download as a zip file. Once you have the zip file, unzip it. 

Download SIMPLE-NN: https://github.com/MDIL-SNU/SIMPLE-NN_v2

----

We encourage you to use ``virtualenv`` or ``conda`` for convenient module managenement when you install SIMPLE-NN.
After downloading the directories, run the command below to install SIMPLE-NN.

.. code-block:: bash

    cd SIMPLE-NN
    python setup.py install

If you run into permission issues, add a ``--user`` tag after the last command.

3. LAMMPS
---------
Currently, we support the module for symmetry_function - Neural_network model.

Download LAMMPS: https://github.com/lammps/lammps

Only LAMMPS whose version is ``29Oct2020`` or later is supported.

Copy the source code to LAMMPS src directory.

.. code-block:: bash

    cp SIMPLE-NN_v2/simple-nn/features/symmetry_function/{pair_nn.*,pair_nn_replica.*} /path/to/lammps/src/
    cp SIMPLE-NN_v2/simple-nn/features/symmetry_function/symmetry_function.h /path/to/lammps/src/

Compile LAMMPS code.

.. code-block:: bash

    cd /path/to/lammps/src/
    make mpi

4. mpi4py (optional)
--------------------
SIMPLE-NN supports the parallel CPU computation in dataset generation and preprocessing for an additional speed gain.

Install mpi4py:

.. code-block:: bash

    pip install mpi4py

-----
Check
-----
To check whether SIMPLE-NN and LAMMPS are ready to run or not,
we provide the shell script in ``installation_check`` directory.

.. note::
    If you use the ``virtualenv`` or ``conda`` for SIMPLE-NN, activate the virtual environment before check.

Run ``run.sh`` with the path of lammps binary.

.. code-block:: bash

    sh run.sh /path/to/lammps/src/lmp_mpi

While ``run.sh`` tests SIMPLE-NN, LAMMPS with neural network potential, and LAMMPS with replica ensemble,
pass or fail messages will be printed like:

.. code-block:: bash
    
    SIMPLE-NN test is passed (or failed).
    LAMMPS with neural network test is passed (or failed).
    LAMMPS with replica ensemble test is passed (or failed).

-----

If you have a problem in installation, post a issues in here_. 

.. _here: https://github.com/MDIL-SNU/SIMPLE-NN_v2/issues
