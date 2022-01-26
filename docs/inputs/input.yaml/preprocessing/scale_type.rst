==========
scale_type
==========

- ``minmax`` (default) / ``meanstd``, ``uniform gas``

----

SIMPLE-NN supports ``minmax``, ``meanstd``, ``uniform gas`` for scaling calculation. The usage of **scal_type** is as below.

.. code-block:: yaml

    # input.yaml
    preprocessing:
        scale_type: minmax

.. note::

    If the **scale_type** tag is set as ``uniform gas``, there is an additional tag named :doc:`/inputs/input.yaml/preprocessing/scale_rho`.
