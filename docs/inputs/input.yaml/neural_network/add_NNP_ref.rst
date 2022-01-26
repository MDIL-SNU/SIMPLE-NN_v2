===========
add_NNP_ref
===========

- ``False`` (default) / ``True``

----

In order to apply replica ensemble to Neural Network Potentials, **add_NNP_ref** must be set as ``True``. Then SIMPLE-NN reads :doc:`/inputs/input.yaml/neural_network/ref_list`, producing atomic energies into the data.pt file. The :doc:`/inputs/input.yaml/neural_network/continue` tag must be set as shown below.

.. code-block:: yaml
    
    #input.yaml
    neural_network:
        continue: weights
