===============
Quick tutorial
===============

Introduction
============

This section demonstrate SIMPLE-NN with tutorials. 
Example files are in ``SIMPLE-NN/tutorials/``.
In this example, snapshots from 500K MD trajectory of 
amorphous SiO\ :sub:`2`\  (72 atoms) are used as training set.  

To run SIMPLE-NN, type the following command: 

.. code-block:: bash

    python run.py

If you have installed ``mpi4py``, MPI parallelization provides an additional speed gain in :ref:`preprocess<preprocess>` (``generate_features`` and ``preprocess`` in ``input.yaml``).

.. code-block:: bash

    mpirun -np $numproc python run.py

where ``numproc`` stands for the number of CPU processors.

.. note::
    In this example, all paths in ``*_list`` such as ``train_list`` and ``valid_list`` are written as relative path.
    Therefore, you should copy ``data`` directory to each example or change the paths properly after the first example :ref:`preprocess`.
     
.. _preprocess:

Preprocess
==========

To preprocess the *ab initio* calculation result for training dataset of NNP, 
you need three types of input file (``input.yaml``, ``structure_list``, and ``params_XX``).
The example files except params_Si and params_O are introduced below.
Detail of params_Si and params_O can be found in :doc:`/inputs/params_XX` section.
In this example, 70 symmetry functions consist of 8 radial symmetry functions per 2-body combination 
and 18 angular symmetry functions per 3-body combination.
Input files introduced in this section can be found in 
``SIMPLE-NN/tutorials/Preprocess``.

.. code-block:: yaml

    # input.yaml
    generate_features: True
    preprocess: True
    train_model: False

    params:
        Si: params_Si
        O: params_O
       
    preprocessing:
        valid_rate: 0.1
        calc_scale: True
        calc_pca: True

.. code-block:: text

    # str_list
    ../ab_initio_output/OUTCAR_comp ::10

With this input file, SIMPLE-NN calculates feature vectors and its derivatives (``generate_features``) and 
generates training/validation dataset (``preprocess``). 
Sample VASP OUTCAR file (the file is compressed to reduce the file size) is in ``SIMPLE-NN/tutorials/ab_initio_output``.

In MD trajectory, snapshots are sampled only in the interval of 10 MD steps (20 fs).

Output files are provided in ``SIMPLE-NN/tutorials/Preprocess_answer`` except for ``data`` directory due to the large capacity.
``data`` directory contains the preprocessed *ab initio* calculation results as binary format named ``data1.pt``, ``data2.pt``, and so on.

If you want to see which data are saved in ``.pt`` file, use the following command. 

.. code-block:: python

    import torch
    result = torch.load('data1.pt')

``result`` provides the information of input features as dictionary format.

.. warning::
    It is recommended to turn on the ``calc_pca`` and ``calc_scale`` options in the ``preprocess``, without which the root-mean-square-error (RMSE) can be high in the ``training``.

.. _training:

Training
========

To train the NNP with the preprocessed dataset, you need to prepare the ``input.yaml``, ``train_list``, ``valid_list``, ``scale_factor``, and ``pca``. The last two files highly improves the loss convergence and training quality.

.. code-block:: yaml

    # input.yaml
    generate_features: False
    preprocess: False
    train_model: True

    params:
        Si: params_Si
        O:  params_O

    neural_network:
        nodes: 30-30
        batch_size: 8
        optimizer: 
            method: Adam
        total_epoch: 100
        learning_rate: 0.001
        use_scale: True
        use_pca: True

With this input file, SIMPLE-NN optimizes the neural network (``train_model``).
The paths of training/validation dataset should be written in ``train_list`` and ``valid_list``, respectively. 
The 70-30-30-1 network is optimized by Adam optimizer with the 0.001 of learning rate and batch size of 8 during 1000 epochs. 
The input feature vectors whose size is 70 are converted by ``scale_factor``, following PCA matrix transformation by ``pca``
The execution log and energy, force, and stress root-mean-squared-error (RMSE) are stored in ``LOG``. 
Input files introduced in this section can be found in ``SIMPLE-NN/tutorials/Training``.


  

.. _evaluation:

Evaluation
==========

To evaluate the training quality of neural network, ``test_list`` and result of training (``checkpoint.pth.tar`` or ``potential_saved``) should be prepared. 
``test_list`` contains the path of testset preprocessed as ``.pt`` format. ``.pt`` format data can be generated as described in :ref:`preprocess<preprocess>`. Note that you should set ``train_list`` to ``test_list`` with ``valid_rate`` of 0.0. Then, SIMPLE-NN will write all paths of preprocessed data in ``test_list``.

.. code-block:: yaml

    # input.yaml
    generate_features: True
    preprocess: True
    train_model: False

    params:
        Si: params_Si
        O: params_O

    preprocessing:
        train_list: 'test_list'
        valid_rate: 0.0
        calc_scale: False
        calc_pca: False
        calc_atomic_weights: False

In this example, ``test_list`` is made by concatenating ``train_list`` and ``valid_list`` in :ref:`training<training>` for simplicity. 
Put the name of result of training such as ``checkpoint_*.tar`` for PyTorch checkpoint file or ``weights`` for LAMMPS potential in ``continue`` in ``input.yaml``. 

.. code-block:: yaml

    # input.yaml
    generate_features: False
    preprocess: False
    train_model: True

    params:
        Si: params_Si
        O:  params_O

    neural_network:
        train: False
        test: True
        continue: checkpoint_bestmodel.pth.tar

Input files introduced in this section can be found in 
``SIMPLE-NN/tutorials/Evaluation``.

.. note::
  If you use LAMMPS potential (``potential_saved``), you need to copy ``pca`` and ``scale_factor`` file and change the name of potential as ``potential_saved``.

After running SIMPLE-NN with the setting above, 
output file named ``test_result`` is generated. 
The file is pickle format and you can open this file with python code of below

.. code-block:: python

    import torch
    result = torch.load('test_result')

In the file, DFT energies/forces, NNP energies/forces are included.
We also provide the python code (``correlation.py``) that makes parity plots from ``test_result``. 

Molecular dynamics
==================

.. note::
  You have to compile your LAMMPS with ``pair_nn.cpp``, ``pair_nn.h``, and ``symmetry_function.h`` to run molecular dynamics simulation.

To run MD simulation with LAMMPS, add the lines into the LAMMPS script file.

.. code-block:: text

    # lammps.in

    units metal

    pair_style nn
    pair_coeff * * /path/to/potential_saved_bestmodel Si O

.. warning::
  This pair_style requires the ``newton`` setting to be ``on(default)`` for pair interactions.

Input script for example of NVT MD simulation at 300 K are provided in ``SIMPLE-NN/tutorials/Molecular dynamics``.
Run LAMMPS via the following command. 

.. code-block:: bash

    /path/to/lammps/src/lmp_mpi < lammps.in

You also can run LAMMPS with ``mpirun`` command if multi-core CPU is supported.

.. code-block:: bash

    mpirun -np $numproc /path/to/lammps/src/lmp_mpi < lammps.in

Output files can be found in ``SIMPLE-NN/tutorials/Molecular_dynamics_answer``.
