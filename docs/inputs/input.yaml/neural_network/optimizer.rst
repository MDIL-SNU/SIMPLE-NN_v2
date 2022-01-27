=========
optimizer
=========

- method: ``Adam`` (default) / ``Adadelta``, ``Adagrad``, ``AdamW``, ``Adamax``, ``ASGD``, ``SGD``, ``RMSprop``, ``Rprop``

----
        
**optimizer** determines the optimization method. The usage of **optimizer** is as below. SIMPLE-NN supports ``Adam``, ``Adadelta``, ``Adagrad``, ``AdamW``, ``Adamax``, ``ASGD``, ``SGD``, ``RMSprop`` and ``Rprop``.

.. code-block:: yaml

    # input.yaml
    neural_network:
        optimizer:
            method: Adam
            params:

As shown below, in general, ``Adam`` type optimizer shows the best convergence.

.. image:: /inputs/input.yaml/neural_network/optimizer.png
   :width: 500
