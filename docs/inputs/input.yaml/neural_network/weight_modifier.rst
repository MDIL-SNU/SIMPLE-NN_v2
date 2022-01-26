===============
weight_modifier
===============

- type: ``null`` (default) / ``modified sigmoid``

----

Dictionary for weight modifier. The usage of **weight_modifier** is as below.

.. code-block:: yaml

    # input.yaml
    neural_network:
        weight_modifier:
            type: modified sigmoid
            params:
