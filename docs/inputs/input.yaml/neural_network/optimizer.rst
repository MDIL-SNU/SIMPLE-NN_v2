=========
optimizer
=========

- method: ``'Adam'`` (default) / ``'Adadelta'``, ``'Adagrad'``, ``'AdamW'``, ``'Adamax'``, ``'ASGD'``, ``'SGD'``, ``'RMSprop'``, ``'Rprop'``
----

.. code-block:: yaml

    # input.yaml
    optimizer:
        method: 'Adam'

        
**Optimizer** determines the optimization method. SIMPLE-NN supports ``'Adam'``, ``'Adadelta'``, ``'Adagrad'``, ``'AdamW'``, ``'Adamax'``, ``'ASGD'``, ``'SGD'``, ``'RMSprop'``, ``'Rprop'``.
