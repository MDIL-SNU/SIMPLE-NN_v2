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
    
5. Intel Accelerator (optional)
-------------------------------

IntelAccelerated version of pair_nn is for molecular dynamics part of
SIMPLE-NN. By exploiting SIMD and MKL’s vector-matrix multiplication
routines, total speed up around 3~3.5 times can be achieved.

Requirements
------------

-  intel CPU supports AVX
-  IntelCompiler ``18.0.5`` or newer
-  IntelMKL ``2018.5.274`` or newer
-  lammps ``23Jun2022-Update1(stable)`` tested

Although only accelerated version of pair_nn requires intel compiler and
lammps itself provides various ways to compile source, We highly
recommend you to compile whole lammps source with intel compiler & intel
mpi(mpiicpc). You can check detailed installation guid for lammps intel
pacakage below.

https://docs.lammps.org/Speed_intel.html

! The code use AVX related functions from intel intrinsic, BLAS routine
and vector mathematics from mkl. So older version of MKL, intel compiler
support those feature would be ok.

Installation
------------

.. code-block:: text

    cp {pair_nn_simd.cpp, pair_nn_simd.h, pair_nn_simd_function.h} {lammps_source}/src/
    cd {lammps_source}/src
    make intel_cpu_intelmpi

Note that intel_cpu_intelmpi is just example for intel compile for lammps.
You may have to change some library path and compile flags if needed. 

Requirements for potential file
-------------------------------

-  Inside symmetry function vector, vector components which share same center atom specie, target atom specie, same symmetry function type form “symmetry function group” 
-  Vector components in same symmetry function group should have same cutoff radius. 
-  Vector components in same symmetry function group should written contiguously in potential file. 
-  For angular symmetry function, zeta should be integer.

Usage
-----
In youer LAMMPS script file,

.. code-block:: text
    # lammps.in
    units metal
    
    pair_style nn/intel
    pair_coeff * * /path/to/potential_saved_bestmodel Si O

See other detailed molecular dynamics docs below
https://simple-nn-v2.readthedocs.io/en/latest/quick_tutorial/quick_tutorial.html#molecular-dynamics


Current Issue
-------------

‘clear’ command inside lammps input script could cause problem.

Further Acceleration
--------------------

If you cpu supports AVX512 instruction set, you can use AVX512 by adding

-D \__AVX512F_\_

to your Makefile’s CCFLAGS. Besides its capacity, speed up respect to
AVX is minor (< 1%). This is because the bottelneck of the accelerated
code is not arithmetic but memory.

SIMD parallelism works on “Symmetry Function Group”. Since AVX
instruction set supports 256bit(total 4 double value) SIMD, setting
number of symmetry function inside one symmetry fuction group multiple
of “4” would results in best speed up w.r.t original version.

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




