========
Examples
========

Introduction
============

This section demonstrate SIMPLE-NN with examples. 
Example files are in :code:`SIMPLE-NN/examples/`.
In this example, snapshots from 500K MD trajectory of 
amorphous SiO\ :sub:`2`\  (72 atoms) are used as training set.  

To run SIMPLE-NN, type the following command on terminal. 

.. code-block:: bash

    python run.py

If you install :code:`mpi4py`, MPI parallelization provides an additional speed gain in :ref:`preprocess` (``generate_features`` and ``preprocess`` in ``input.yaml``).

.. code-block:: bash

    mpirun -np numproc python run.py

, where ``numproc`` stands for the number of CPU processors.

.. _preprocess:

1. Preprocess
=============

To preprocess the *ab initio* calculation result for training dataset of NNP, 
you need three types of input file (:code:`input.yaml`, :code:`structure_list`, and :code:`params_XX`).
The example files except params_Si and params_O are introduced below.
Detail of params_Si and params_O can be found in :doc:`/features/symmetry_function/symmetry_function` section.
In this example, 70 symmetry functions consist of 8 radial symmetry functions per 2-body combination 
and 18 angular symmetry functions per 3-body combination.
Input files introduced in this section can be found in 
:code:`SIMPLE-NN/examples/1.Preprocess`.

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

.. code-block:: bash

    # str_list
    ../ab_initio_output/OUTCAR_comp ::10

With this input file, SIMPLE-NN calculates feature vectors and its derivatives (:code:`generate_features`) and 
generates training/validation dataset (:code:`preprocess`). 
Sample VASP OUTCAR file (the file is compressed to reduce the file size) is in :code:`SIMPLE-NN/examples/ab_initio_output`.

In MD trajectory, snapshots are sampled only in the interval of 10 MD steps (20 fs).

Output files are provided in :code:`SIMPLE-NN/examples/1.Preprocess_answer` except for ``data`` directory due to the large capacity.
``data`` directory contains the preprocessed *ab initio* calculation results as binary format named ``data1.pt``, ``data2.pt``, and so on.

If you want to see which data are saved in ``.pt`` file, use the following command. 

.. code-block:: python

    import torch
    result = torch.load('data1.pt')

``result`` provides the information of input features as dictionary format.

.. _training:

2. Training
===========

To train the NNP with the preprocessed dataset, you need to prepare the :code:`input.yaml`, :code:`train_list`, :code:`valid_list`, :code:`scale_factor`, and :code:`pca`. The last two files highly improves the loss convergence and training quality.

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
        scale: True
        pca: True

.. note::
    You should check the path in ``train_list`` and ``valid_list``. For this example, copy the ``data`` directory from ``1.Preprocess`` to here or change the paths in ``train_list`` and ``valid_list`` from ``./data/data*.pt`` to ``../1.Preprocess/data/data*.pt``
     
With this input file, SIMPLE-NN optimizes the neural network (:code:`train_model`).
The paths of training/validation dataset should be written in :code:`train_list` and :code:`valid_list`, respectively. 
The 70-30-30-1 network is optimized by Adam optimizer with the 0.001 of learning rate and batch size of 8 during 1000 epochs. 
The input feature vectors whose size is 70 are converted by :code:`scale_factor`, following PCA matrix transformation by :code:`pca`
The execution log and energy, force, and stress root-mean-squared-error (RMSE) are stored in :code:`LOG`. 
Input files introduced in this section can be found in :code:`SIMPLE-NN/examples/2.Training`.

3. Evaluation
=============

To evaluate the quality of training by correlation between reference dataset and NNP as well as RMSE, :code:`test_list` should be prepared. 
:code:`test_list` contains the path of testset preprocessed as '.pt' format. 
In this example, :code:`test_list` is made by concatenating :code:`train_list` and :code:`valid_list` in :ref:`training` for simplicity. 
Testset in :code:`test_list` also can be generated separately as described in :code:`1. Preprocess`. 
In this case, we recommend you to run :ref:`preprocess` with ``valid_rate`` of 0.0 and then change the filename of :code:`train_list` into :code:`test_list`. 
The potential to be tested is written in ``continue``. Both :code:`checkpoint.tar` and :code:`potential_saved` can be used when evaluation.

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
:code:`SIMPLE-NN/examples/3.Evaluation`.

.. note::
  You need to copy :code:`pca` and :code:`scale_factor` files if you use LAMMPS potential (``potential_saved``). 

After running SIMPLE-NN with the setting above, 
output file named :code:`test_result` is generated. 
The file is pickle format and you can open this file with python code of below

.. code-block:: python

    import torch
    result = torch.load('test_result')

In the file, DFT energies/forces, NNP energies/forces are included.
We also provide the python code (:code:`correlation.py`) that makes parity plots from :code:`test_result`. 

4. Molecular dynamics
=====================
To run MD simulation with LAMMPS, add the lines into the LAMMPS script file.

.. code-block:: bash
    # lammps.in

    units metal

    pair_style nn
    pair_coeff * * /path/to/potential_saved_bestmodel Si O

Input script for example of NVT MD simulation at 300 K are provided in :code:`SIMPLE-NN/example/4.Molecular dynamics`.
Run LAMMPS via the following command. You also can run LAMMPS with ``mpirun`` command if multi-core CPU is supported.

.. code-block:: bash

    /path/to/lammps/src/lmp_mpi < lammps.in

Output files can be found in :code:`SIMPLE-NN/examples/4.Molecular_dynamics_answer`.

5. Parameter tuning (GDF)
=========================

GDF [#f1]_ is used to reduce the force errors of the sparsely sampled atoms. 
To use GDF, you need to calculate the :math:`\rho(\mathbf{G})` 
by adding the following lines to the :code:`symmetry_function` section in :code:`input.yaml`.
SIMPLE-NN supports automatic parameter generation scheme for :math:`\sigma` and :math:`c`.
Use the setting :code:`sigma: Auto` to get a robust :math:`\sigma` and :math:`c` (values are stored in LOG file).
Input files introduced in this section can be found in 
:code:`SIMPLE-NN/examples/SiO2/parameter_tuning_GDF`.

::

    #symmetry_function:
      #continue: true # if individual pickle file is not deleted
      atomic_weights:
        type: gdf
        params:
          sigma: Auto
          # for manual setting
          #  Si: 0.02 
          #  O: 0.02


:math:`\rho(\mathbf{G})` indicates the density of each training point.
After calculating :math:`\rho(\mathbf{G})`, histograms of :math:`\rho(\mathbf{G})^{-1}` 
are also saved as in the file of :code:`GDFinv_hist_XX.pdf`.

.. note::
  If there is a peak in high :math:`\rho(\mathbf{G})^{-1}` region in the histogram, 
  increasing the Gaussian weight(:math:`\sigma`) is recommended until the peak is removed.
  On the contrary, if multiple peaks are shown in low :math:`\rho(\mathbf{G})^{-1}` region in the histogram,
  reduce :math:`\sigma` is recommended until the peaks are combined. 

In the default setting, the group of :math:`\rho(\mathbf{G})^{-1}` is scaled to have average value of 1. 
The interval-averaged force error with respect to the :math:`\rho(\mathbf{G})^{-1}` 
can be visualized with the following script.


::

    from simple_nn.utils import graph as grp

    grp.plot_error_vs_gdfinv(['Si','O'], 'test_result')

The graph of interval-averaged force errors with respect to the 
:math:`\rho(\mathbf{G})^{-1}` is generated as :code:`ferror_vs_GDFinv_XX.pdf`

If default GDF is not sufficient to reduce the force error of sparsely sampled training points, 
One can use scale function to increase the effect of GDF. In scale function, 
:math:`b` controls the decaying rate for low :math:`\rho(\mathbf{G})^{-1}` and 
:math:`c` separates highly concentrated and sparsely sampled training points.
To use the scale function, add following lines to the :code:`symmetry_function` section in :code:`input.yaml`.

::

    #symmetry_function:
      weight_modifier:
        type: modified sigmoid
        params:
          Si:
            b: 0.02
            c: 3500.
          O:
            b: 0.02
            c: 10000.

For our experience, :math:`b=1.0` and automatically selected :math:`c` shows reasonable results. 
To check the effect of scale function, use the following script for visualizing the 
force error distribution according to :math:`\rho(\mathbf{G})^{-1}`. 
In the script below, :code:`test_result_noscale` is the test result file from the training without scale function and 
:code:`test_result_wscale` is the test result file from the training with scale function.

::

    from simple_nn.utils import graph as grp

    grp.plot_error_vs_gdfinv(['Si','O'], 'test_result_noscale', 'test_result_wscale')




.. [#f1] `W. Jeong, K. Lee, D. Yoo, D. Lee and S. Han, J. Phys. Chem. C 122 (2018) 22790`_

.. _W. Jeong, K. Lee, D. Yoo, D. Lee and S. Han, J. Phys. Chem. C 122 (2018) 22790: https://pubs.acs.org/doi/abs/10.1021/acs.jpcc.8b08063

6. Uncertainty estimation
=========================

.. note::
  Before this step, you have to compile your LAMMPS with :code:`pair_nn_replica.cpp` and :code:`pair_nn_replica.h`.

LAMMPS can calculate the atomic uncertainty through standard deviation of atomic energies.
Because our NNP do not deal with charged system, atomic uncertainty can be written as atomic charge.
Prepare your data file as charge format and please modify your LAMMPS input as below example.

::

    atom_style  charge
    pair_style  nn/r (# of replica potentials)
    pair_coeff  * * (reference potential) (element1) (element2) ... &
                (replica potential_#1) &
                (replica_potential_#2) &
                ...
    compute     (ID) (group-ID) property/atom q

.. [#f2] `W. Jeong, D. Yoo, K. Lee, J. Jung and S. Han, J. Phys. Chem. Lett. 2020, 11, 6090-6096`_

.. _W. Jeong, D. Yoo, K. Lee, J. Jung and S. Han, J. Phys. Chem. Lett. 2020, 11, 6090-6096: https://pubs.acs.org/doi/10.1021/acs.jpclett.0c01614

