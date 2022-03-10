=================
Advanced features
=================

Introduction
============

This section demonstrate SIMPLE-NN with tutorials. 
Example files are in ``SIMPLE-NN/tutorials/``.
In this example, snapshots from 500K MD trajectory of 
amorphous SiO\ :sub:`2`\  (72 atoms) are used as training set.  

GDF weighting
=============

Tuning the weight of atomic force in loss function can be used to reduce the force errors of the sprasely sampled atoms.
Gaussian densigy function (GDF) weighting [#f1]_ is one of the methods, which suggests the gaussian type of weighting scheme. 
To use GDF, you need to calculate the :math:`\rho(\mathbf{G})` 
by adding the following lines to the ``symmetry_function`` section in ``input.yaml``.
SIMPLE-NN supports automatic parameter generation scheme for :math:`\sigma` and :math:`c`.
Use the setting ``sigma: Auto`` to get a robust :math:`\sigma` and :math:`c` (values are stored in LOG file).
Input files introduced in this section can be found in 
``SIMPLE-NN/tutorials/GDF_weighting``.

.. code-block:: yaml

    # input.yaml:

    preprocessing:
        valid_rate: 0.1
        calc_scale: True
        calc_pca: True
        calc_atomic_weights:
            type: gdf
            params: Auto

:math:`\rho(\mathbf{G})` indicates the density of each training point.
After calculating :math:`\rho(\mathbf{G})`, histograms of :math:`\rho(\mathbf{G})^{-1}` 
are also saved as in the file of ``GDFinv_hist_XX.pdf``.

.. note::
  If there is a peak in high :math:`\rho(\mathbf{G})^{-1}` region in the histogram, 
  increasing the Gaussian weight(:math:`\sigma`) is recommended until the peak is removed.
  On the contrary, if multiple peaks are shown in low :math:`\rho(\mathbf{G})^{-1}` region in the histogram,
  reduce :math:`\sigma` is recommended until the peaks are combined. 

In the default setting, the group of :math:`\rho(\mathbf{G})^{-1}` is scaled to have average value of 1. 
The interval-averaged force error with respect to the :math:`\rho(\mathbf{G})^{-1}` 
can be visualized with the following script.

.. code-block:: python

    from simple_nn.utils import graph as grp
    grp.plot_error_vs_gdfinv(['Si','O'], 'test_result')

The graph of interval-averaged force errors with respect to the 
:math:`\rho(\mathbf{G})^{-1}` is generated as ``ferror_vs_GDFinv_XX.pdf``

If default GDF is not sufficient to reduce the force error of sparsely sampled training points, 
One can use scale function to increase the effect of GDF. In scale function, 
:math:`b` controls the decaying rate for low :math:`\rho(\mathbf{G})^{-1}` and 
:math:`c` separates highly concentrated and sparsely sampled training points.
To use the scale function, add following lines to the ``neural_network`` section in ``input.yaml``.

.. code-block:: yaml

    # input.yaml:
    
    neural_network:
        weight_modifier:
            type: modified sigmoid
            params:
                Si:
                    b: 1
                    c: 35.
                O:
                    b: 1
                    c: 74.

For our experience, :math:`b=1.0` and automatically selected :math:`c` shows reasonable results. 
To check the effect of scale function, use the following script for visualizing the 
force error distribution according to :math:`\rho(\mathbf{G})^{-1}`. 

In the script below, ``test_result_woscale`` is the test result file from the training without scale function and 
``test_result_wscale`` is the test result file from the training with scale function.
These ``test_result`` are made as described in :ref:`evaluation<evaluation>`. We do not provide ``test_result_wscale``.

.. code-block:: python

    from simple_nn.utils import graph as grp
    grp.plot_error_vs_gdfinv(['Si','O'], 'test_result_woscale', 'test_result_wscale')

.. [#f1] `W. Jeong, K. Lee, D. Yoo, D. Lee and S. Han, J. Phys. Chem. C 122 (2018) 22790`_

.. _W. Jeong, K. Lee, D. Yoo, D. Lee and S. Han, J. Phys. Chem. C 122 (2018) 22790: https://pubs.acs.org/doi/abs/10.1021/acs.jpcc.8b08063

Uncertainty estimation
======================

The local configuration shown in the simulation driven by NNP should be included the training set because NNP only guarantees the reliability within the trained domain.
Therefore, we suggest to check whether the local environment is trained or not through the standard deviation of atomic energies from replica ensemble [#f2]_.
To estimate the uncertainty of atomic configuration, following three steps are needed. 

.. _atomic_energy_extraction:

1. Atomic energy extraction
---------------------------

To estimatet the uncertainty of atomic configuration, the atomic energies extracted from reference NNP should be added into reference dataset (``.pt``).

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
        test: False
        add_NNP_ref: True
        ref_list: 'ref_list'
        train_atomic_E: False
        scale: True
        pca: True
        continue: checkpoint_bestmodel.pth.tar
    
``ref_list`` contains the dataset list to be evaluated to atomic energy. Reference NNP is written in ``continue``.
After that, the reference dataset (``.pt``) are overwritten with atomic energies.

2. Training with atomic energy
------------------------------ 

Next, train the replica NNP only with atomic energy.
To prevent the convergence among replicas,
diversity the network structure by increasing the standard deviation of initial weight distribution (``gain`` (default: 1.0)) and change the number of hidden nodes larger than reference NNP.

.. code-block:: yaml

    # input.yaml

    generate_features: False
    preprocess: False
    train_model: True
    random_seed: 123

    params:
        Si: params_Si
        O:  params_O

    neural_network:
        train: False
        test: False
        add_NNP_ref: False
        train_atomic_E: True
        nodes: 30-30
        weight_initializer:
            params:
                gain: 2.0  
        optimizer:
            method: Adam
        total_epoch: 100
        learning_rate: 0.001
        scale: True
        pca: True
        continue: null

Because the atomic energies are needed in training, ``data`` directory made from :ref:`atomic_energy_extraction<atomic_energy_extraction>` is needed.

3. Uncertainty estimation in molecular dynamics
-----------------------------------------------

.. note::
  You have to compile your LAMMPS with ``pair_nn_replica.cpp``, ``pair_nn_replica.h``, and ``symmetry_function.h`` to evaluate the uncertainty in molecular dynamics simulation.

LAMMPS can calculate the atomic uncertainty through standard deviation of atomic energies.
Because atomic uncertainty will be written as atomic charge,
prepare LAMMPS data file as charge format and modify your LAMMPS input as below example.

.. code-block:: bash
    
    # lammps.in

    units       metal
    atom_style  charge

    pair_style  nn/r 3
    pair_coeff  * * potential_saved Si O &
                potential_saved_30 &
                potential_saved_60 &
                potential_saved_90 

    compute     std all property/atom q

    dump        mydump all custom 1 dump.lammps id type x y z c_std
    dump_modify sort id

    run 1

We provide the LAMMPS potentials whose network size are 60-60 and 90-90, respectively.
Atomic uncertainties are written in a dump file for each atoms.
Outputs files are found in ``SIMPLE-NN/tutorials/Uncertainty_estimation_answer/3.Uncertainty_estimation_in_molecular_dynamics``.

.. [#f2] `W. Jeong, D. Yoo, K. Lee, J. Jung and S. Han, J. Phys. Chem. Lett. 2020, 11, 6090-6096`_

.. _W. Jeong, D. Yoo, K. Lee, J. Jung and S. Han, J. Phys. Chem. Lett. 2020, 11, 6090-6096: https://pubs.acs.org/doi/10.1021/acs.jpclett.0c01614

