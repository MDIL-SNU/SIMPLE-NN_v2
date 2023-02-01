==================
weight_initializer
==================

- type: ``xavier normal`` (default) / ``xavier uniform``, ``normal``, ``constant``, ``kaiming normal``, ``kaiming uniform``, ``he normal``, ``he uniform``, ``orthogonal``, ``sparse``

----

Weight initialization is used to define the initial values for the parameters in Neural Network models prior to training the models on dataset. SIMPLE-NN supports several **weight_initializer** and the usage of **weight_initializer** is as below.

.. code-block:: yaml

    # input.yaml
    neural_network:
        weight_initializer:
            type: xavier normal
            params: 
