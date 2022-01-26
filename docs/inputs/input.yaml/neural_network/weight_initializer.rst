==================
weight_initializer
==================

- type: ``xavier noraml`` (default) / ``xavier uniform``, ``normal``, ``constant``, ``kaiming noraml``, ``kaiming uniform``, ``he normal``, ``he uniform``, ``orthogonal``, ``sparse``

----

The usage of **weight_initializer** is as below.

.. code-block:: yaml

    # input.yaml
    neural_network:
        weight_initializer:
            type: xavier noraml
