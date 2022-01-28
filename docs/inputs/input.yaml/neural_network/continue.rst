========
continue
========

- ``null`` (default) / ``weights``, ``checkpoint_bestmodel.pth.tar``

----

If the **continue** tag is set to ``weights``, the training process restarts from the LAMMPS potential file(``potential_saved``). If the tag is set to ``checkpoint_bestmodel.pth.tar``, the training process restarts from the checkpoint file. The usage of **continue** is as below.

.. code-block:: yaml
    
    #input.yaml
    neural_network:
        continue: weights / checkpoint_bestmodel.pth.tar

.. note::

   Users need to copy ``pca`` and ``scale_factor`` and potential files if you use LAMMPS potential(change the name of potential file into ``potential_saved``).

   Users need to copy ``checkpoint_bestmodel.pth.tar`` into your running directory if you use checkpoint file.
