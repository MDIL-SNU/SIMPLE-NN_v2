.. _install:

============
Installation
============

------------
Requirements
------------
- Python :code:`3.6-3.9`
- LAMMPS :code:`29Oct2020` or later

1. Pytorch
----------
Install PyTorch: https://pytorch.org/get-started/locally

Choose the PyTorch of stable release for `Python`. If you have CUDA-capable system, please download PyTorch with CUDA that makes training much faster.

To check if your GPU driver and CUDA are enabled by PyTorch, run the following commands in python to return whether or not the CUDA driver is enabled: 
.. code_block:: python3

    import torch.cuda
    torch.cuda.is_available()

2. SIMPLE-NN
------------
We encourage you to use `virtualenv` or `conda` for convenient module managenement.
.. code_block:: bash

    cd SIMPLE-NN_v2
    python setup.py install

If you run into permission issues, add a `--user` tag after the last command.
2-1. From github url
====================
.. code_block:: bash

    git clone https://github.com/MDIL-SNU/SIMPLE-NN_v2.git

2-2. From source file
=====================
You can download a current SIMPLE-NN source package from link below. 
Once you have a zip file, unzip it. This will create SIMPLE-NN directory.
After unzipping the file, run the command below to install SIMPLE-NN.

Download SIMPLE-NN: https://github.com/MDIL-SNU/SIMPLE-NN_v2

3. LAMMPS
---------
Currently, we support the module for symmetry_function - Neural_network model.

Download LAMMPS: https://github.com/lammps/lammps

Only LAMMPS whose version is `29Oct2020` or later is supported.

Copy the source code to LAMMPS src directory.
.. code_block:: bash

    cp SIMPLE-NN_v2/simple-nn/features/symmetry_function/pair_nn.* /path/to/lammps/src/
    cp SIMPLE-NN_v2/simple-nn/features/symmetry_function/symmetry_function.h /path/to/lammps/src/

Compile LAMMPS code.
.. code_block:: bash

    cd /path/to/lammps/src/
    make mpi

4. mpi4py (optional)
--------------------
SIMPLE-NN supports the parallel CPU computation in dataset generation and preprocessing for an additional speed gain.

Install mpi4py:
.. code_block:: bash

    pip install mpi4py
