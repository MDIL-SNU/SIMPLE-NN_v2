===========
Random_seed
===========

- ``null`` (default) / non-negative integer

----

**random_seed** is used to set the seed of random number generator in SIMPLE-NN. SIMPLE-NN has randomness in train/valid separation, data loading, and weight initialization. When ``random_seed`` is set to ``null``, SIMPLE-NN generates the random number based on your system time. You can reproduce the same training result with the random seed value written the top of ``LOG`` file.

