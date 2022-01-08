======
Params
======

- Type: ``dict``

**params** contains the path of parameter files for each atom.
For example, when the system consists of Si and O atoms, params should be written down like this:

.. code-block:: yaml

    params:
        Si: params_Si
        O: params_O

The detailed description of ``params_XX`` can be found in :doc:`/inputs/params_XX`.
The order of species determines the index in ``params_XX``.
