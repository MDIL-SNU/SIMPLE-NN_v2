==============
dx_save_sparse
==============

- ``True`` (default) / ``False``

----

**dx_save_sparse** determines whether the derivative of input feature matrix, which is used to calculate force from atomic energy in training process, is saved as sparse or dense tensor. Generally, sparse tensor has smaller capacity but provides slower training speed. We recommend testing on your system before setting. It only works when :doc:`/inputs/input.yaml/data/read_force` is ``True``.
