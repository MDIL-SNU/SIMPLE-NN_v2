=========
use_force
=========

- ``True`` (default) / ``False``

----

If the **use_force** tag is set as ``True``, force is used for training.  From our experience, we recommend training with both energy and forces for robust Neural Network Potential, since training with only energy induces overfitting, while training with forces only gives large errors in total energy.
