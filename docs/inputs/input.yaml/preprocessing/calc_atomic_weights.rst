===================
calc_atomic_weights
===================

- ``False`` (default) / ``gdf``

----

As mentioned in :doc:`/advanced_features/advanced_features` section, tuning the weight of atomic force in loss function can be used to reduce the force errors of sparsely sampled atoms. In order to activate atomic weights, the usage of **calc_atomic_weights** is shown as below. SIMPLE-NN supports automatic parameter generation scheme for :math:`\sigma` and :math:`c`. Use the setting ``params: Auto`` to get a robust :math:`\sigma` and :math:`c`.


.. code-block:: yaml

    # input.yaml
    preprocessing:
        calc_atomic_weights:
            type: gdf
            params: Auto
