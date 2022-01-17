===
FAQ
===

* How to mitigate the overfittng?

    * Try dropout of ``True`` , larger l2_regularization, less the number of node.

* How to restart from previous training?

    * Write the file name of ``checkpoint`` of ``potential_saved`` in ``neural_network`` of ``input.yaml``. Do not forget to copy ``scale_factor`` and ``pca`` when using ``potential_saved``.
