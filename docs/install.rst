.. _install:

============
Installation
============

------------
Requirements
------------
- Python ``3.6-3.9``
- PyTorch ``1.5.0-1.10.1`` (package for machine learning)
- LAMMPS ``23Jun2022`` or newer (simulator for molecular dynamics)


Optional:

- mpi4py (library for parallel CPU computation in preprocessing)

----

----------
Procedures
----------

1. Pytorch
----------
Install PyTorch: https://pytorch.org/get-started/locally

Choose the PyTorch of stable release for ``Python``. If you have CUDA-capable system, please download PyTorch with CUDA that makes training much faster.

To check if your GPU driver and CUDA are enabled by PyTorch, run the following commands in python to return whether or not the CUDA driver is enabled: 

.. code-block:: python

    import torch.cuda
    torch.cuda.is_available()

2. SIMPLE-NN
------------

2-1. Download via git clone
===========================
You can download a SIMPLE-NN source code through cloning from repository like this:

.. code-block:: text

    git clone https://github.com/MDIL-SNU/SIMPLE-NN_v2.git SIMPLE-NN

2-2. Download as a zip file
===========================
Alternatively, you can download a current SIMPLE-NN source code as zip file from link below. 

Download SIMPLE-NN: https://github.com/MDIL-SNU/SIMPLE-NN_v2

----

.. note::
    We recommend using ``venv`` or ``conda`` for convenient module managenement.

After downloading the SIMPLE-NN, install SIMPLE-NN with following the command.

.. code-block:: text

    cd SIMPLE-NN
    python setup.py install

If you run into permission issues, add a ``--user`` tag after the last command.

3. LAMMPS
---------
Currently, we support the module for symmetry_function - Neural_network model.

Download LAMMPS: https://github.com/lammps/lammps

Only LAMMPS whose version is ``23Jun2022`` or newer is supported.

Copy the source code to LAMMPS src directory.

.. code-block:: text


    cp SIMPLE-NN_v2/simple-nn/features/symmetry_function/pair_nn* /path/to/lammps/src/
    cp SIMPLE-NN_v2/simple-nn/features/symmetry_function/symmetry_function.h /path/to/lammps/src/


pair_nn* in the first command includes the ``pair_nn.cpp``, ``pair_nn.h``, ``pair_nn_replica.cpp``, and ``pair_nn_replica.h``.

Compile LAMMPS code.

.. code-block:: text

    cd /path/to/lammps/src/
    make mpi

After this step, you can :ref:`test your installation<test_installation>`. 

4. mpi4py (optional)
--------------------
SIMPLE-NN supports the parallel CPU computation in dataset generation and preprocessing for an additional speed gain.

Install mpi4py:

.. code-block:: text

    pip install mpi4py

.. _test_installation:

----------------------
Test your installation
----------------------
To check whether SIMPLE-NN and LAMMPS are ready to run or not,
we provide the shell script in ``test_installation`` directory.

.. note::
    If you use the ``venv`` or ``conda`` for SIMPLE-NN, activate the virtual environment before check.

Run ``run.sh`` with the path of lammps binary.

.. code-block:: text

    ./run.sh /path/to/lammps/src/lmp_mpi

While ``run.sh`` tests SIMPLE-NN, LAMMPS with neural network potential, and LAMMPS with replica ensemble,
pass or fail messages will be printed like:

.. code-block:: text
    
    Test is going on...
    SIMPLE-NN test is passed (or failed).
    LAMMPS with neural network test is passed (or failed).
    LAMMPS with replica ensemble test is passed (or failed).

-----

If you have a problem in installation, post a issues in here_. 

.. _here: https://github.com/MDIL-SNU/SIMPLE-NN_v2/issues
