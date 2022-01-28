====================
clear_prev_optimizer
====================

- ``False`` (default) / ``True``

----

**clear_prev_optimizer** determines whether SIMPLE-NN continues with the optimizer inside the ``checkpoint_bestmodel.pth.tar`` file(``False``)or not.

.. code-block:: yaml
    
    #input.yaml
    neural_network:
        continue: checkpoint_bestmodel.pth.tar
    clear_prev_optimizer: False

.. note::
   More details of **clear_prev_optimizer** is under construction.
