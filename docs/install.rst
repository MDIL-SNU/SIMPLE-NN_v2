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
    
5. Intel SIMD acceleration (optional)
-------------------------------

The filename extension simd refers to Intel-accelerated version of simulating molecular dynamics in SIMPLE-NN. By utilizing  vector-matrix multiplication routines in Intel MKL and vectorizing descriptor computation by SIMD, overall speed up would be x3 to x3.5 faster than the regular version.

5.1 Requirements
================

-  Intel CPU supporting AVX
-  Compiler supporting AVX instruction set
-  IntelMKL ``2018.5.274`` tested
-  LAMMPS ``23Jun2022-Update1(stable)`` tested

In our experience, the best performance is achieved when source compiled with intel compiler(icpc) and intel mpi (mpiicpc). LAMMPS provides default makefile for intel compiler, intel mpi and mkl library path setting. Therefore, we recommend to compile lammps source with intel compiler.

The code uses AVX-related functions from intel intrinsic,  BLAS routines of MKL, and vector math. So if older versions of MKL and intel compilers support these features, there is no problem for compiling.

5.2 Installation
================

.. code-block:: text

    cp {simple_nn_path}/simple_nn/features/symmetry_function/SIMD/{pair_nn_simd.cpp, pair_nn_simd.h, pair_nn_simd_function.h} {lammps_source}/src/
    cd {lammps_source}/src
    make intel_cpu_intelmpi
    
.. note::
    'make intel_cpu_intelmpi' is an example of using the intel compiler for lammps. Before using a makefile, you may need to explicitly set some library path and optimization flags (such as -xAVX) in the makefile if necessary.

5.3 Requirements for potential file
===================================
-  Symmetry function group refers to a group of vector components which have the same target atom specie(s). 
-  Vector components of the same symmetry function group must have the same cutoff radius.
-  Vector components of the same symmetry function group must be contiguous in potential file.
-  The zeta value must be an integer in the angular symmetry functions.

Since some assumptions have been made about the potential files for acceleration, the potential file must follow the rules above.

5.4 Usage
=========
In youer LAMMPS script file, regular version uses ``pair_style nn``.
For the accelerated version, ``pair_style nn/intel`` should be invoked.

5.5 Further Acceleration
========================
Two additional accelerations are possible if the AVX2 or AVX512 instruction set is available.
To enable these features, add "-xCORE-AVX2" or "-xCORE-AVX512" compile flag to your makefile, depending on your CPU.
Since AVX512 is released after AVX2, turning on AVX512 automatically turns on AVX2 as well.

Further acceleration by AVX2 is possible by computing unique values of symmetry function parameters to reduce computation.
So it puts some requirements on potential file.
 - The potential file must contain at least one G4 or G5 angular symmetry function.
 - The number of unique 'eta' value in same angular symmetry function group must be less than 4(AVX2) or 8(AVX512).
 - The zeta value must be less than 8.
This acceleration is about 25~35% faster than the primitive AVX version.

In addition, AVX512 doubles the maximum size of simd calculation, whose speed up is around 10%.

You can check the log file of LAMMPS to see if the installation was successful and if the potential file conditions were met.
After LAMMPS reads the potential file, you can see somthing like this :

.. code-block:: text

    AVX2 for angular descriptor G4 calc : on/off
    AVX2 for angular descriptor G5 calc : on/off
    AVX512 for descriptor calc : on/off

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




