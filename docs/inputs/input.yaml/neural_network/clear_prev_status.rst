=================
clear_prev_status
=================

- ``False`` (default) / ``True``

----

**clear_prev_status** determines whether SIMPLE-NN continues from the :doc:/`/inputs.input.yaml/neural_network/start_epoch` with the corresponding network inside the ``checkpoint_bestmodel.pth.tar`` file(``False``)or not. The usage of **clear_prev_status** is shown as below.

.. code-block:: yaml
    
    #input.yaml
    neural_network:
        continue: checkpoint_bestmodel.pth.tar
    clear_prev_status: True
    start_epoch: 5

.. note::
   More details of **clear_prev_optimizer** is under construction.
