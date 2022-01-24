=========
optimizer
=========

- method: ``'Adam'`` (default) / ``'Adadelta'``, ``'Adagrad'``, ``'AdamW'``, ``'Adamax'``, ``'ASGD'``, ``'SGD'``, ``'RMSprop'``, ``'Rprop'``
----
        
**optimizer** determines the optimization method. The usage of **optimizer** is as below. SIMPLE-NN supports ``'Adam'``, ``'Adadelta'``, ``'Adagrad'``, ``'AdamW'``, ``'Adamax'``, ``'ASGD'``, ``'SGD'``, ``'RMSprop'``, ``'Rprop'``.

.. code-block:: yaml

    # input.yaml
    optimizer:
        method: 'Adam'
