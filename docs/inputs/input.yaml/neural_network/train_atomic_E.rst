==============
train_atomic_E
==============

- ``False`` (default) / ``True``

----

If the **train_atomic_E** tag is set as ``True``, based on the train_list and valid_list which were produced from :doc:`/inputs/input.yaml/neural_network/add_NNP_ref` step, SIMPLE-NN trains one set of replica ensemble. By varying weight parameters and network size, users can apply replica ensemble to Neural Network Potentials. The :doc:`/inputs/input.yaml/neural_network/continue` tag must be set as shown below.

.. code-block:: yaml
    
    #input.yaml
    neural_network:
        continue: null
