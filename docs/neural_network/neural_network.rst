==============
Neural network
==============

In this section, you can find the information of neural network in SIMPLE-NN.

Running mode
============

.. toctree::
    :maxdepth: 1

    train
    train_list
    valid_list 
    test
    test_list
    add_NNP_ref
    ref_list
    train_atomic_E
    use_force
    use_stress
    shuffle_dataloader

Network
=======

.. toctree::
    :maxdepth: 1

    nodes
    acti_func
    double_precision
    weight_initializer
    dropout
    scale
    pca
    atomic_weights
    weight_modifier

Optimization
============

.. toctree::
    :maxdepth: 1

    optimizer
    batch_size
    full_batch
    total_epoch
    learning_rate
    decay_rate
    l2_regularization

Loss function
=============

.. toctree::
    :maxdepth: 1

    loss_scale
    E_loss_type
    F_loss_type
    energy_coeff
    force_coeff 
    stress_coeff

Logging & saving
================

.. toctree::
    :maxdepth: 1

    show_interval
    save_interval
    energy_criteria
    force_criteria
    stress_criteria
    print_structure_rmse

Continue
========

.. toctree::
    :maxdepth: 1

    continue
    clear_prev_status
    clear_prev_optimizer
    start_epoch

Parallelism
===========

.. toctree::
    :maxdepth: 1

    use_gpu        
    GPU_number     
    inter_op_threads
    intra_op_threads
    load_data_to_gpue
    subprocesses
