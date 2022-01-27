==========
calc_scale
==========

- ``True`` (default) / ``False``

----

**calc_scale** determines whether SIMPLE-NN calculates scaling parameters(``True``) or not(``False``). Feature scaling is a method used to normalize the range of independent variables or features of data. It is required because as the range of raw data varies widely, the range of all features should be normalized in order to match the contribution of each feature proportionately to the final width. SIMPLE-NN supports several scaling method as described in :doc:`/inputs/input.yaml/preprocessing/scale_type`.
